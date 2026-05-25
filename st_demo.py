from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import streamlit as st
from streamlit_echarts import st_echarts

from embedding import Embeddings
from main import (
    Node,
    Record,
    SearchResult,
    SemanticBTree,
    build_tree_from_records,
    validate_record,
)


class _EChartsItem(TypedDict):
    name: str
    value: int | None


class _EChartsNode(TypedDict):
    name: str
    collapsed: bool
    children: list[_EChartsNode | _EChartsItem]


def build_node_label(name: str, size: int, radius: float) -> str:
    """Return a formatted multiline label for an ECharts tree node."""
    return f"{name}\nsize={size}\nradius={radius:.3f}"


def build_root_label(node: Node) -> str:
    """Return a formatted label for the root node."""
    return (
        f"{node.node_id}\n"
        f"size={node.size}\n"
        f"radius={node.radius:.3f}\n"
        f"max_id={node.max_item_id}"
    )


def build_leaf_label(text: str) -> str:
    """Return a multiline label for a leaf item."""
    return text


def tree_to_echarts(
    node: Node,
    depth: int = 0,
    is_root: bool = False,
) -> _EChartsNode:
    """Convert a semantic tree node into ECharts tree data."""
    label = (
        build_root_label(node)
        if is_root
        else build_node_label(
            name=node.node_id,
            size=node.size,
            radius=node.radius,
        )
    )

    if node.is_leaf:
        children: list[_EChartsNode | _EChartsItem] = [
            _EChartsItem(
                name=build_leaf_label(
                    text=item.payload.text if item.payload is not None else ""
                ),
                value=item.item_id,
            )
            for item in node.items
        ]
        return _EChartsNode(name=label, collapsed=depth >= 2, children=children)

    return _EChartsNode(
        name=label,
        collapsed=depth >= 2,
        children=[tree_to_echarts(child, depth=depth + 1) for child in node.children],
    )


def render_tree(tree: SemanticBTree) -> None:
    """Render the semantic tree using ECharts."""
    data = tree_to_echarts(tree.root, is_root=True)

    options = {
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove",
        },
        "series": [
            {
                "type": "tree",
                "data": [data],
                "top": "2%",
                "left": "10%",
                "bottom": "2%",
                "right": "28%",
                "symbol": "circle",
                "symbolSize": 10,
                "orient": "LR",
                "layout": "orthogonal",
                "expandAndCollapse": True,
                "initialTreeDepth": -1,
                "edgeShape": "polyline",
                "edgeForkPosition": "50%",
                "roam": True,
                "lineStyle": {
                    "width": 1.5,
                    "curveness": 0,
                },
                "label": {
                    "position": "left",
                    "verticalAlign": "middle",
                    "align": "right",
                    "fontSize": 12,
                    "lineHeight": 18,
                    "backgroundColor": "#f5f7fb",
                    "borderColor": "#d9e1ec",
                    "borderWidth": 1,
                    "borderRadius": 6,
                    "padding": [6, 8],
                },
                "leaves": {
                    "label": {
                        "position": "right",
                        "verticalAlign": "middle",
                        "align": "left",
                        "lineHeight": 18,
                        "backgroundColor": "#fff8e8",
                        "borderColor": "#f0d7a1",
                        "borderWidth": 1,
                        "borderRadius": 6,
                        "padding": [6, 8],
                    }
                },
                "animationDuration": 300,
                "animationDurationUpdate": 500,
            }
        ],
    }

    st_echarts(options=options, height="900px")


def load_records(file_path: Path) -> list[Record]:
    """Load the initial records for the demo."""
    with file_path.open("r", encoding="utf-8") as file:
        raw: list[object] = json.load(file)
    return [validate_record(entry) for entry in raw]


def initialize_records(file_path: Path) -> None:
    """Initialize editable records in session state."""
    if "records" not in st.session_state:
        st.session_state.records = load_records(file_path)


def invalidate_tree() -> None:
    """Drop cached tree metadata from session state."""
    st.session_state.pop("tree", None)
    st.session_state.pop("tree_key", None)


def get_next_record_id() -> int:
    """Return the next available record id."""
    tree: SemanticBTree | None = st.session_state.get("tree")

    if tree is not None and tree.root.max_item_id is not None:
        return tree.root.max_item_id + 1

    records: list[Record] = st.session_state.records
    numeric_ids = [record.id for record in records]
    return max(numeric_ids, default=0) + 1


def add_record(text: str) -> tuple[bool, str]:
    """Add a new record into the editable dataset."""
    normalized_text = text.strip()

    if not normalized_text:
        return False, "Заполните text."

    next_record_id = get_next_record_id()
    records: list[Record] = st.session_state.records
    st.session_state.records = [
        *records,
        Record(id=next_record_id, text=normalized_text),
    ]
    invalidate_tree()
    return True, f"Запись с id={next_record_id} добавлена."


def delete_record(record_id: str) -> tuple[bool, str]:
    """Delete a record by id from the editable dataset."""
    normalized_id = record_id.strip()

    if not normalized_id:
        return False, "Укажите id для удаления."

    records: list[Record] = st.session_state.records
    filtered_records = [record for record in records if str(record.id) != normalized_id]

    if len(filtered_records) == len(records):
        return False, f"Запись с id={normalized_id} не найдена."

    st.session_state.records = filtered_records
    invalidate_tree()
    return True, f"Запись с id={normalized_id} удалена."


def find_item_path(tree: SemanticBTree, item_id: int) -> list[str] | None:
    """Return the node path from root to the leaf containing the given item id."""

    def dfs(node: Node, path: list[str]) -> list[str] | None:
        """Traverse the tree and return the path when the item is found."""
        current_path = [*path, node.node_id]

        if node.is_leaf:
            if any(str(item.item_id) == str(item_id) for item in node.items):
                return current_path
            return None

        for child in node.children:
            child_path = dfs(child, current_path)
            if child_path is not None:
                return child_path

        return None

    return dfs(tree.root, [])


def render_feedback(tree: SemanticBTree | None) -> None:
    """Render success or error feedback for the last mutation."""
    pending_added_id: int | None = st.session_state.pop("pending_added_id", None)

    if pending_added_id is not None and tree is not None:
        path = find_item_path(tree=tree, item_id=pending_added_id)
        if path is not None:
            st.session_state.feedback = (
                "success",
                f"Запись с id={pending_added_id} добавлена. Ветка: {' -> '.join(path)}",
            )
        else:
            st.session_state.feedback = (
                "success",
                f"Запись с id={pending_added_id} добавлена.",
            )

    feedback: tuple[str, str] | None = st.session_state.pop("feedback", None)
    if feedback is not None:
        level, message = feedback
        if level == "success":
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)


def get_or_build_tree(
    records: list[Record],
    embedder: Embeddings,
    t_value: int,
) -> SemanticBTree | None:
    """Return a tree from session state or rebuild it when parameters change."""
    if not records:
        invalidate_tree()
        return None

    records_signature = tuple(str(record.id) for record in records)
    current_key = f"tree_t_{t_value}_{records_signature}"

    if st.session_state.get("tree_key") != current_key:
        st.session_state.tree = build_tree_from_records(
            records=records,
            embeddings=embedder,
            t=t_value,
        )
        st.session_state.tree_key = current_key

    return st.session_state.tree


def render_record_controls() -> None:
    """Render forms for adding and deleting records."""
    with st.sidebar.expander("Управление узлами", expanded=True):
        with st.form("add_record_form", clear_on_submit=True):
            st.markdown("**Добавить узел**")
            text = st.text_area("text", key="add_record_text", height=120)
            add_submitted = st.form_submit_button("Добавить")

            if add_submitted:
                success, message = add_record(text=text)
                if success:
                    added_id = message.split("id=")[1].split()[0]
                    st.session_state.pending_added_id = int(added_id)
                    st.rerun()
                else:
                    st.session_state.feedback = ("error", message)

        with st.form("delete_record_form", clear_on_submit=True):
            st.markdown("**Удалить узел по id**")
            record_id_to_delete = st.text_input(
                "id для удаления",
                key="delete_record_id",
            )
            delete_submitted = st.form_submit_button("Удалить")

            if delete_submitted:
                success, message = delete_record(record_id=record_id_to_delete)
                if success:
                    st.session_state.feedback = ("success", message)
                    st.rerun()
                else:
                    st.session_state.feedback = ("error", message)


def render_records_overview(records: list[Record]) -> None:
    """Render a compact overview of current record identifiers."""
    st.sidebar.caption(f"Всего записей: {len(records)}")
    st.sidebar.caption(
        f"Следующий id: {get_next_record_id()}" if records else "Следующий id: 1"
    )
    st.sidebar.caption(
        "Текущие id: " + ", ".join(str(record.id) for record in records[:12])
    )

    if len(records) > 12:
        st.sidebar.caption("Показаны первые 12 id.")


def render_search_results(
    tree: SemanticBTree | None,
    embedder: Embeddings,
    query: str,
    k_value: int,
) -> None:
    """Render search results for the current query."""
    if tree is None:
        st.info("Нет данных для поиска. Добавьте хотя бы одну запись.")
        return

    if st.button("Искать"):
        query_vector = embedder.embed(query)
        results: list[SearchResult] = tree.search(query_vector, k=k_value)

        st.subheader("Результаты")

        for result in results:
            if result.payload is not None:
                st.write(f"**id={result.payload.id}**")
                st.write(result.payload.text)
            st.caption(
                f"distance={result.distance:.4f} | path={' -> '.join(result.path)}"
            )


def render_tree_panel(tree: SemanticBTree | None) -> None:
    """Render the tree panel or an empty-state message."""
    st.subheader("Дерево")

    if tree is None:
        st.info("Дерево пустое. Добавьте записи, чтобы построить его снова.")
        return

    render_tree(tree)


def main() -> None:
    """Run the Streamlit demo application."""
    st.set_page_config(layout="wide")
    st.title("Semantic B-Tree Demo")

    data_path = Path(__file__).with_name("initial_data.json")
    initialize_records(data_path)
    embedder = Embeddings()

    t_value = st.sidebar.slider("B-tree t", 2, 50, 8)
    k_value = st.sidebar.slider("Top-K", 1, 20, 5)
    render_record_controls()

    records: list[Record] = st.session_state.records
    render_records_overview(records)

    tree = get_or_build_tree(records=records, embedder=embedder, t_value=t_value)
    render_feedback(tree)

    query = st.text_input(
        "Введите поисковый запрос",
        "космический корабль и спутники",
    )

    left, right = st.columns([1, 2])

    with left:
        render_search_results(
            tree=tree,
            embedder=embedder,
            query=query,
            k_value=k_value,
        )

    with right:
        render_tree_panel(tree)


if __name__ == "__main__":
    main()
