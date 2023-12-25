import matplotlib.pyplot as plt
from dataclasses import dataclass
import random
import networkx as nx
import json
import os
import pandas as pd


class TreeNode:
    def __init__(self, value : str):
        self.value = value
        self.left = None
        self.right = None


@dataclass
class Person:
    name: str
    income: int
    credit_score: int
    debt_to_income_ratio: float
    employment_type: str
    years_in_current_job: int
    open_credit_lines: int
    loan_amount: float
    delinquent_accounts: int
    result: str = None

    def __post_init__(self):
        if self.income > 50000:
            if self.credit_score > 700:
                if self.debt_to_income_ratio < 30:
                    self.result = "Approve"
                elif self.employment_type == "Permanent":
                    self.result = "Approve"
                else:
                    self.result = "Review"
            else:
                if self.years_in_current_job > 5:
                    self.result = "Approve"
                elif self.open_credit_lines > 3:
                    self.result = "Reject"
                else:
                    self.result = "Review"
        else:
            if self.loan_amount < 100000:
                if self.credit_score > 650:
                    self.result = "Approve"
                else:
                    self.result = "Review"
            else:
                if self.years_in_current_job > 3:
                    if self.delinquent_accounts == 0:
                        self.result = "Approve"
                    else:
                        self.result = "Reject"
                else:
                    self.result = "Reject"

    def to_dict(self):
        return self.__dict__


def generate_person() -> Person:
    names = [
        "Иван",
        "Мария",
        "Александр",
        "Екатерина",
        "Дмитрий",
        "Анна",
        "Сергей",
        "Ольга",
        "Николай",
        "Татьяна",
    ]
    surnames = [
        "Иванов",
        "Петров",
        "Сидоров",
        "Козлова",
        "Михайлов",
        "Новиков",
        "Алексеев",
        "Кузнецова",
        "Лебедев",
        "Семенов",
    ]

    return Person(
        name=f"{random.choices(names)[0]} {random.choices(surnames)[0]}",
        income=random.randint(25000, 75000),
        credit_score=random.randint(500, 900),
        debt_to_income_ratio=random.randint(15, 45),
        employment_type=random.choice(["Permament", "Part-Time"]),
        years_in_current_job=random.randint(2, 8),
        open_credit_lines=random.randint(0, 6),
        loan_amount=random.randint(50000, 250000),
        delinquent_accounts=random.randint(0, 2),
    )


def json_to_tree(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def build_tree(node_data: dict, layer=1, order=1) -> TreeNode:
    node = TreeNode(f"{node_data['node']} ({layer},{order})")
    if "left_leaf" in node_data:
        node.left = build_tree(node_data["left_leaf"], layer + 1, order * 2 - 1)
    if "right_leaf" in node_data:
        node.right = build_tree(node_data["right_leaf"], layer + 1, order * 2)

    return node


def add_edges(graph: nx.DiGraph, node: TreeNode, pos=None, x=0, y=0, layer=1) -> None:
    if pos is None:
        pos = {node.value: (x, y)}

    if node.left is not None:
        label = "Yes"
        graph.add_edge(node.value, node.left.value, label=label)
        pos[node.left.value] = (x - 1 / 2**layer, y - 1)
        add_edges(graph, node.left, pos, x - 1 / 2**layer, y - 1, layer + 1)

    if node.right is not None:
        label = "No"
        graph.add_edge(node.value, node.right.value, label=label)
        pos[node.right.value] = (x + 1 / 2**layer, y - 1)
        add_edges(graph, node.right, pos, x + 1 / 2**layer, y - 1, layer + 1)


def visualize_tree(node: TreeNode, save_path=None) -> None:
    graph = nx.DiGraph()
    pos = {node.value: (0, 0)}
    add_edges(graph, node, pos)

    # масштабирование
    pos_higher = {k: (v[0] * 2, v[1] * 2) for k, v in pos.items()}
    labels = {k: k.split(" (")[0] for k in pos.keys()}
    edge_labels = nx.get_edge_attributes(graph, "label")

    fig, ax = plt.subplots(figsize=(20, 8))
    nx.draw(
        graph,
        pos=pos_higher,
        labels=labels,
        with_labels=True,
        arrows=True,
        node_size=700,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        ax=ax,
    )
    nx.draw_networkx_edge_labels(graph, pos=pos_higher, edge_labels=edge_labels)

    plt.savefig(save_path)


if __name__ == '__main__':
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree.json")
    tree = json_to_tree(filepath)
    root = build_tree(tree)
    visualize_tree(root, save_path="tree_visualization.png")

    borrowers = [generate_person() for i in range(10)]
    print(pd.DataFrame(borrowers))
