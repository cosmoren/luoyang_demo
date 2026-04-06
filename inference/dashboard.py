#!/usr/bin/env python
"""Dash dashboard: two images on the left, refreshed on an interval."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import traceback
from pathlib import Path

from dash import Dash, Input, Output, dcc, html


def image_path_upper() -> Path | str:
    """Return filesystem path to the upper image on the left."""
    return "/home/cosmo/workspace/skimg/asi16/asi_16613/20260403/20260403164100_12.jpg"


def image_path_lower() -> Path | str:
    """Return filesystem path to the lower image on the left."""
    return "/home/cosmo/workspace/skimg/asi16/asi_16613/20260403/20260403164100_12.jpg"


def image_src_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


IMG_CLASS = "dash-panel-img"
IMG_STYLE = {
    "display": "block",
    "borderRadius": "8px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.08)",
}


def _panel_content(get_path, label: str):
    """Build children for one image panel; tolerates missing files and NotImplemented."""
    try:
        p = Path(get_path()).resolve()
    except NotImplementedError as e:
        return html.Div(
            str(e) or f"Implement {get_path.__name__}()",
            style={
                "color": "#8b4049",
                "padding": "8px",
                "fontSize": "14px",
                "maxHeight": "100%",
                "overflow": "auto",
                "boxSizing": "border-box",
            },
        )
    except Exception as e:
        return html.Div(
            [html.Strong(f"{label}: "), html.Pre(traceback.format_exc()[-800:])],
            style={
                "color": "#8b4049",
                "fontSize": "12px",
                "maxHeight": "100%",
                "overflow": "auto",
                "boxSizing": "border-box",
            },
        )

    if not p.is_file():
        return html.Div(
            f"Waiting for file: {p}",
            style={
                "color": "#5c6370",
                "padding": "12px",
                "fontSize": "14px",
                "maxHeight": "100%",
                "overflow": "auto",
                "boxSizing": "border-box",
            },
        )

    try:
        src = image_src_data_uri(p)
    except OSError as e:
        return html.Div(f"Cannot read {p}: {e}", style={"color": "#8b4049", "padding": "8px"})

    return html.Div(
        html.Img(src=src, alt=str(p), className=IMG_CLASS, style=IMG_STYLE),
        className="dash-square-frame",
    )


def build_app(refresh_interval_ms: int) -> Dash:
    app = Dash(__name__)
    app.title = "Inference dashboard"

    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body { height: 100%; margin: 0; overflow: hidden; }
            #react-entry-point { height: 100%; }
            #react-entry-point > div { height: 100%; }
            ._dash-app-content { height: 100%; overflow: hidden; box-sizing: border-box; }
            body { font-family: system-ui, sans-serif; background: #e8eaee; color: #1a1a1a; }
            .dashboard-root { height: 100%; box-sizing: border-box; }
            .image-panel {
                flex: 0 0 auto;
                align-self: flex-start;
                width: fit-content;
                max-width: 100%;
                box-sizing: border-box;
                background: #fff;
                padding: 6px;
                border-radius: 10px;
                border: 1px solid #d8dce3;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            /* Same square size for both images; cap so they stay small on large screens */
            .dash-square-frame {
                width: min(18vmin, 152px);
                height: min(18vmin, 152px);
                aspect-ratio: 1;
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .dash-panel-img {
                width: 100%;
                height: 100%;
                object-fit: contain;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

    app.layout = html.Div(
        [
            html.Div(
                dcc.Interval(
                    id="refresh-interval",
                    interval=max(500, refresh_interval_ms),
                    n_intervals=0,
                ),
                style={"height": "0", "overflow": "hidden", "flexShrink": 0},
            ),
            html.Div(
                [
                    html.H1(
                        "太阳能功率预测实时监控",
                        style={
                            "margin": 0,
                            "fontSize": "1.1rem",
                            "fontWeight": "600",
                            "lineHeight": "1.2",
                            "flexShrink": 0,
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(id="panel-upper", className="image-panel"),
                                    html.Div(id="panel-lower", className="image-panel"),
                                ],
                                style={
                                    "flex": "0 0 auto",
                                    "minWidth": "0",
                                    "maxWidth": "52%",
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "alignItems": "flex-start",
                                    "gap": "10px",
                                },
                            ),
                            html.Div(style={"flex": "1", "minWidth": "0", "minHeight": "0"}),
                        ],
                        style={
                            "flex": "1",
                            "minHeight": "0",
                            "display": "flex",
                            "flexDirection": "row",
                            "alignItems": "flex-start",
                            "gap": "16px",
                            "overflow": "hidden",
                        },
                    ),
                ],
                style={
                    "flex": "1",
                    "minHeight": "0",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "10px",
                    "padding": "12px 16px",
                    "boxSizing": "border-box",
                    "overflow": "hidden",
                },
            ),
        ],
        className="dashboard-root",
        style={
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
            "boxSizing": "border-box",
        },
    )

    @app.callback(
        Output("panel-upper", "children"),
        Output("panel-lower", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def refresh_images(_n: int):
        return _panel_content(image_path_upper, "Upper"), _panel_content(image_path_lower, "Lower")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dash: two images on the left, paths from image_path_upper / image_path_lower; auto-refresh."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument(
        "--interval",
        type=int,
        default=30000, # 30 seconds
        metavar="MS",
        help="Refresh interval in milliseconds (min 500)",
    )
    parser.add_argument("--debug", action="store_true", help="Dash debug mode")
    args = parser.parse_args()

    app = build_app(args.interval)
    print(f"Open http://{args.host}:{args.port}/  (refresh every {max(500, args.interval)} ms)")
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
