from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import Directive


class PlotlyExecDirective(Directive):
    """Execute embedded Python and render Plotly figure HTML.

    The directive expects code content that defines a variable named ``fig``
    (a Plotly figure object).
    """

    has_content = True
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        code = "\n".join(self.content)
        if not code.strip():
            return [nodes.paragraph(text="No code provided for plotly-exec.")]

        namespace: dict[str, object] = {}
        try:
            exec(code, namespace)
        except Exception as exc:  # pragma: no cover - docs-time diagnostic path
            message = nodes.paragraph(
                text=f"plotly-exec failed while running code: {exc!r}"
            )
            return [message]

        fig = namespace.get("fig")
        if fig is None:
            return [nodes.paragraph(text="plotly-exec requires a `fig` variable.")]

        try:
            html = fig.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config={"displaylogo": False, "responsive": True},
            )
        except Exception as exc:  # pragma: no cover - docs-time diagnostic path
            message = nodes.paragraph(
                text=f"plotly-exec could not serialize figure: {exc!r}"
            )
            return [message]

        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_directive("plotly-exec", PlotlyExecDirective)
    return {"version": "0.1", "parallel_read_safe": True}
