import asyncio
import logging
import os
from concurrent.futures import Executor, ProcessPoolExecutor
from datetime import datetime, timezone
from functools import cache, partial
from multiprocessing import freeze_support
from typing import Any

try:
    from aiohttp import web
    from multidict import MultiMapping

    from .middlewares import cors
except ImportError as ie:
    raise ImportError(
        f"aiohttp dependency is not installed: {ie}. "
        + "Please re-install black with the '[d]' extra install "
        + "to obtain aiohttp_cors: `pip install black[d]`"
    ) from None

import click

import black
from _black_version import version as __version__
from black.concurrency import maybe_use_uvloop
from black.files import _load_toml, find_pyproject_toml

# This is used internally by tests to shut down the server prematurely
_stop_signal = asyncio.Event()

# Request headers
PROTOCOL_VERSION_HEADER = "X-Protocol-Version"
LINE_LENGTH_HEADER = "X-Line-Length"
PYTHON_VARIANT_HEADER = "X-Python-Variant"
SKIP_SOURCE_FIRST_LINE = "X-Skip-Source-First-Line"
SKIP_STRING_NORMALIZATION_HEADER = "X-Skip-String-Normalization"
SKIP_MAGIC_TRAILING_COMMA = "X-Skip-Magic-Trailing-Comma"
PREVIEW = "X-Preview"
UNSTABLE = "X-Unstable"
ENABLE_UNSTABLE_FEATURE = "X-Enable-Unstable-Feature"
FAST_OR_SAFE_HEADER = "X-Fast-Or-Safe"
DIFF_HEADER = "X-Diff"

BLACK_HEADERS = [
    PROTOCOL_VERSION_HEADER,
    LINE_LENGTH_HEADER,
    PYTHON_VARIANT_HEADER,
    SKIP_SOURCE_FIRST_LINE,
    SKIP_STRING_NORMALIZATION_HEADER,
    SKIP_MAGIC_TRAILING_COMMA,
    PREVIEW,
    UNSTABLE,
    ENABLE_UNSTABLE_FEATURE,
    FAST_OR_SAFE_HEADER,
    DIFF_HEADER,
]

# Response headers
BLACK_VERSION_HEADER = "X-Black-Version"
DEFAULT_MAX_BODY_SIZE = 5 * 1024 * 1024
DEFAULT_WORKERS = os.cpu_count() or 1


class HeaderError(Exception):
    pass


class InvalidVariantHeader(Exception):
    pass


def parse_pyproject_toml(path_config: str) -> dict[str, Any]:
    """Parse a pyproject toml file, pulling out the `[tool.blackd]` section.

    Keys are normalized the same way as for Black's own `[tool.black]` section.
    If parsing fails, will raise a tomllib.TOMLDecodeError.
    """
    pyproject_toml = _load_toml(path_config)
    config: dict[str, Any] = pyproject_toml.get("tool", {}).get("blackd", {})
    return {k.replace("--", "").replace("-", "_"): v for k, v in config.items()}


def read_pyproject_toml(
    ctx: click.Context, param: click.Parameter | None, value: str | None
) -> str | None:
    """Inject blackd configuration from "pyproject.toml" into defaults in `ctx`.

    The file is looked up the same way Black looks up its own configuration,
    starting from the current working directory.

    Returns the path to a successfully found and read configuration file, None
    otherwise.
    """
    if not value:
        value = find_pyproject_toml((), None)
        if value is None:
            return None

    try:
        config = parse_pyproject_toml(value)
    except (OSError, ValueError) as e:
        raise click.FileError(
            filename=value, hint=f"Error reading configuration file: {e}"
        ) from None

    if not config:
        return None

    # Config keys are the long option names, which do not always match the name of
    # the parameter (e.g. --cors-allow-origin is stored in cors_allow_origins).
    option_to_param = {
        opt[2:].replace("-", "_"): p.name
        for p in ctx.command.params
        for opt in p.opts
        if opt.startswith("--") and p.name is not None
    }
    config = {option_to_param.get(k, k): v for k, v in config.items()}
    black.spellcheck_pyproject_toml_keys(ctx, list(config), value)

    cors_allow_origins = config.get("cors_allow_origins")
    if cors_allow_origins is not None and not isinstance(cors_allow_origins, list):
        raise click.BadOptionUsage(
            "cors-allow-origin", "Config key cors-allow-origin must be a list"
        )

    # Sanitize the values to be Click friendly, like Black does.
    config = {
        k: str(v) if not isinstance(v, (list, dict)) else v for k, v in config.items()
    }

    default_map: dict[str, Any] = {}
    if ctx.default_map:
        default_map.update(ctx.default_map)
    default_map.update(config)

    ctx.default_map = default_map
    return value


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        allow_dash=False,
        path_type=str,
    ),
    is_eager=True,
    callback=read_pyproject_toml,
    help=(
        "Read configuration options from a configuration file. By default the"
        " [tool.blackd] section of the pyproject.toml found the same way as Black's"
        " configuration is used."
    ),
)
@click.option(
    "--bind-host",
    type=str,
    help="Address to bind the server to.",
    default="localhost",
    show_default=True,
)
@click.option(
    "--bind-port", type=int, help="Port to listen on", default=45484, show_default=True
)
@click.option(
    "--cors-allow-origin",
    "cors_allow_origins",
    multiple=True,
    help="Origin allowed to access blackd over CORS. Can be passed multiple times.",
)
@click.option(
    "--max-body-size",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_BODY_SIZE,
    show_default=True,
    help="Maximum request body size in bytes.",
)
@click.version_option(version=black.__version__)
def main(
    config: str | None,
    bind_host: str,
    bind_port: int,
    cors_allow_origins: tuple[str, ...],
    max_body_size: int,
) -> None:
    logging.basicConfig(level=logging.INFO)
    if config:
        black.out(f"Using configuration from {config}.")
    app = make_app(cors_allow_origins=cors_allow_origins, max_body_size=max_body_size)
    ver = black.__version__
    black.out(f"blackd version {ver} listening on {bind_host} port {bind_port}")
    loop = maybe_use_uvloop()
    try:
        web.run_app(
            app,
            host=bind_host,
            port=bind_port,
            handle_signals=True,
            print=None,
            loop=loop,
        )
    finally:
        if not loop.is_closed():
            loop.close()


@cache
def executor() -> Executor:
    return ProcessPoolExecutor(max_workers=DEFAULT_WORKERS)


def make_app(
    *,
    cors_allow_origins: tuple[str, ...] = (),
    max_body_size: int = DEFAULT_MAX_BODY_SIZE,
) -> web.Application:
    app = web.Application(
        client_max_size=max_body_size,
        middlewares=[
            cors(
                allow_headers=(*BLACK_HEADERS, "Content-Type"),
                allow_origins=frozenset(cors_allow_origins),
                expose_headers=(BLACK_VERSION_HEADER,),
            )
        ],
    )
    app.add_routes([
        web.post(
            "/",
            partial(
                handle,
                executor=executor(),
                executor_semaphore=asyncio.BoundedSemaphore(DEFAULT_WORKERS),
            ),
        )
    ])
    return app


async def handle(
    request: web.Request,
    executor: Executor,
    executor_semaphore: asyncio.BoundedSemaphore,
) -> web.Response:
    headers = {BLACK_VERSION_HEADER: __version__}
    try:
        if request.headers.get(PROTOCOL_VERSION_HEADER, "1") != "1":
            return web.Response(
                status=501, text="This server only supports protocol version 1"
            )

        fast = False
        if request.headers.get(FAST_OR_SAFE_HEADER, "safe") == "fast":
            fast = True
        try:
            mode = parse_mode(request.headers)
        except HeaderError as e:
            return web.Response(status=400, text=e.args[0])
        req_bytes = await request.read()
        charset = request.charset if request.charset is not None else "utf8"
        req_str = req_bytes.decode(charset)
        then = datetime.now(timezone.utc)

        header = ""
        if mode.skip_source_first_line:
            first_newline_position: int = req_str.find("\n") + 1
            header = req_str[:first_newline_position]
            req_str = req_str[first_newline_position:]

        only_diff = bool(request.headers.get(DIFF_HEADER, False))
        formatted_str = await format_code(
            req_str=req_str,
            fast=fast,
            mode=mode,
            then=then,
            only_diff=only_diff,
            executor=executor,
            executor_semaphore=executor_semaphore,
        )

        # Put the source first line back
        req_str = header + req_str
        formatted_str = header + formatted_str

        return web.Response(
            content_type=request.content_type,
            charset=charset,
            headers=headers,
            text=formatted_str,
        )
    except black.NothingChanged:
        return web.Response(status=204, headers=headers)
    except black.InvalidInput as e:
        return web.Response(status=400, headers=headers, text=str(e))
    except black.SourceASTParseError as e:
        return web.Response(status=400, headers=headers, text=str(e))
    except web.HTTPException:
        raise
    except Exception as e:
        logging.exception("Exception during handling a request")
        return web.Response(status=500, headers=headers, text=str(e))


async def format_code(
    *,
    req_str: str,
    fast: bool,
    mode: black.FileMode,
    then: datetime,
    only_diff: bool,
    executor: Executor,
    executor_semaphore: asyncio.BoundedSemaphore,
) -> str:
    async with executor_semaphore:
        loop = asyncio.get_event_loop()
        formatted_str = await loop.run_in_executor(
            executor, partial(black.format_file_contents, req_str, fast=fast, mode=mode)
        )

        if not only_diff:
            return formatted_str

        now = datetime.now(timezone.utc)
        src_name = f"In\t{then}"
        dst_name = f"Out\t{now}"
        return await loop.run_in_executor(
            executor,
            partial(black.diff, req_str, formatted_str, src_name, dst_name),
        )


def parse_mode(headers: MultiMapping[str]) -> black.Mode:
    try:
        line_length = int(headers.get(LINE_LENGTH_HEADER, black.DEFAULT_LINE_LENGTH))
    except ValueError:
        raise HeaderError("Invalid line length header value") from None

    if PYTHON_VARIANT_HEADER in headers:
        value = headers[PYTHON_VARIANT_HEADER]
        try:
            pyi, versions = parse_python_variant_header(value)
        except InvalidVariantHeader as e:
            raise HeaderError(
                f"Invalid value for {PYTHON_VARIANT_HEADER}: {e.args[0]}",
            ) from None
    else:
        pyi = False
        versions = set()

    skip_string_normalization = bool(
        headers.get(SKIP_STRING_NORMALIZATION_HEADER, False)
    )
    skip_magic_trailing_comma = bool(headers.get(SKIP_MAGIC_TRAILING_COMMA, False))
    skip_source_first_line = bool(headers.get(SKIP_SOURCE_FIRST_LINE, False))

    preview = bool(headers.get(PREVIEW, False))
    unstable = bool(headers.get(UNSTABLE, False))
    enable_features: set[black.Preview] = set()
    enable_unstable_features = headers.get(ENABLE_UNSTABLE_FEATURE, "").split(",")
    for piece in enable_unstable_features:
        piece = piece.strip()
        if piece:
            try:
                enable_features.add(black.Preview[piece])
            except KeyError:
                raise HeaderError(
                    f"Invalid value for {ENABLE_UNSTABLE_FEATURE}: {piece}",
                ) from None

    return black.FileMode(
        target_versions=versions,
        is_pyi=pyi,
        line_length=line_length,
        skip_source_first_line=skip_source_first_line,
        string_normalization=not skip_string_normalization,
        magic_trailing_comma=not skip_magic_trailing_comma,
        preview=preview,
        unstable=unstable,
        enabled_features=enable_features,
    )


def parse_python_variant_header(value: str) -> tuple[bool, set[black.TargetVersion]]:
    value = value.strip()
    if value == "pyi":
        return True, set()
    else:
        versions = set()
        for version in value.split(","):
            version = version.strip()
            if version.startswith("py"):
                version = version[len("py") :]
            if not version:
                raise InvalidVariantHeader("expected e.g. '3.7', 'py3.5'")
            if "." in version:
                major_str, *rest = version.split(".")
            else:
                major_str = version[0]
                rest = [version[1:]] if len(version) > 1 else []
            try:
                major = int(major_str)
                if major not in (2, 3):
                    raise InvalidVariantHeader("major version must be 2 or 3")
                if len(rest) > 0:
                    minor = int(rest[0])
                    if major == 2:
                        raise InvalidVariantHeader("Python 2 is not supported")
                else:
                    # Default to lowest supported minor version.
                    minor = 7 if major == 2 else 3
                version_str = f"PY{major}{minor}"
                if major == 3 and not hasattr(black.TargetVersion, version_str):
                    raise InvalidVariantHeader(f"3.{minor} is not supported")
                versions.add(black.TargetVersion[version_str])
            except (KeyError, ValueError):
                raise InvalidVariantHeader("expected e.g. '3.7', 'py3.5'") from None
        return False, versions


def patched_main() -> None:
    freeze_support()
    main()


if __name__ == "__main__":
    patched_main()
