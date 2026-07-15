"""统一终端输出：基于 rich 的 Console 与常用美化助手。

用法：
    from utils.console import console, step, summary_table
    console.print("[green]OK[/]")
    step("1. 初始化数据库")
"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def step(title: str):
    """打印一个步骤分隔标题。"""
    console.rule(f"[bold cyan]{title}[/]")


def banner(title: str, subtitle: str = ""):
    """打印一个带副标题的面板横幅。"""
    body = Text(subtitle, style="dim") if subtitle else None
    console.print(Panel(Text(title, style="bold white"), subtitle=body))


def make_table(title: str, columns: list) -> Table:
    """创建一张统一风格的表格。

    columns: [(标题, 对齐方式), ...]，对齐默认 "left"。
    """
    t = Table(title=title, title_style="bold magenta", header_style="bold")
    for col in columns:
        name = col[0]
        align = col[1] if len(col) > 1 else "left"
        t.add_column(name, justify=align)
    return t
