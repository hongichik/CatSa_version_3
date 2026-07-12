"""Giai đoạn 2 — Module 1 phần A: xây heterogeneous session graph.

session_to_graph(phiên) → HeteroData với:
    Node : item (các item trong phiên), category (leaf), parent (nếu có taxonomy)
    Edge : sequential (item→item), membership (item→category),
           taxonomy (category→parent)
    + các cạnh ngược (rev_*) để message passing hai chiều — nhờ đó thông tin
    category/parent chảy NGƯỢC về item embedding (readout chỉ dùng item node).
Graph là LOCAL: mỗi phiên một graph riêng; khi train gom B graph bằng
Batch.from_data_list.

Giới hạn phạm vi (finding G2, CatSA_Correctness_Synthesis): cat_parent là
dict[int,int] — mỗi category CHỈ có 1 parent (cây/tree hoặc flat), KHÔNG hỗ
trợ DAG đa-cha. Đây là quyết định phạm vi có chủ đích cho CatSA/Q3 — cả
RetailRocket (taxonomy dạng cây) và Diginetica/Yoochoose (flat, không
taxonomy) đều không cần đa-cha. Tổng quát hoá sang DAG (dict[int,list[int]],
mỗi parent 1 taxonomy edge riêng) chỉ cần thiết khi mở rộng sang CAPro-CL/Q1
với Tmall (đa cấp) hoặc MIND.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, HeteroData


def session_to_graph(
    session: list[int],
    item2cat: dict[int, int],
    cat_parent: dict[int, int] | None = None,
    use_taxonomy: bool = True,
    add_star_node: bool = False,
) -> HeteroData:
    """Chuyển một phiên (list item index) thành heterogeneous graph HeteroData.

    add_star_node: thêm 1 "session" node ảo (SGNN-HN, Pan et al. 2020) nối 2
    chiều với mọi item node — cho phép thông tin lan truyền xa (item đầu ↔
    item cuối) chỉ qua 2 hop, không cần tăng số lớp GNN (tránh over-smoothing).
    Mặc định false — không đổi graph của các phiên bản trước.
    """
    # has_taxonomy tính MỘT LẦN, dùng thống nhất mọi nơi (finding G1/G4):
    # trước đây node/edge "parent" được tạo chỉ dựa vào use_taxonomy, không
    # kiểm tra cat_parent — dataset flat (Yoochoose/Diginetica) gọi nhầm với
    # use_taxonomy=True + cat_parent=None sẽ tạo node type "parent" RỖNG và
    # taxonomy edge (2,0) một cách âm thầm (có thể crash/NaN tùy version PyG).
    # Đồng thời, một khi dataset CÓ taxonomy (has_taxonomy=True), "parent"
    # LUÔN được tạo cho MỌI phiên (dù rỗng nếu phiên đó không có parent nào)
    # để schema nhất quán giữa các phiên trong cùng dataset.
    has_taxonomy = use_taxonomy and bool(cat_parent)
    assert len(session) >= 1, "session_to_graph cần phiên non-empty (>= 1 item)"

    g = HeteroData()

    # --- Node: unique item / category / parent (giữ thứ tự xuất hiện) ---
    item_nodes: list[int] = []
    item_local: dict[int, int] = {}
    for it in session:
        if it not in item_local:
            item_local[it] = len(item_nodes)
            item_nodes.append(it)

    cat_nodes: list[int] = []
    cat_local: dict[int, int] = {}
    for it in item_nodes:
        if it not in item2cat:
            raise KeyError(
                f"item {it} không có trong item2cat — vocabulary/lookup thiếu "
                "category cho item này (kiểm tra tienxuly preprocess)."
            )
        c = item2cat[it]
        if c not in cat_local:
            cat_local[c] = len(cat_nodes)
            cat_nodes.append(c)

    parent_nodes: list[int] = []
    parent_local: dict[int, int] = {}
    if has_taxonomy:
        for c in cat_nodes:
            p = cat_parent.get(c)
            if p is not None and p not in parent_local:
                parent_local[p] = len(parent_nodes)
                parent_nodes.append(p)

    # node_id = index toàn cục để lookup embedding table trong encoder
    g["item"].node_id = torch.tensor(item_nodes, dtype=torch.long)
    g["item"].num_nodes = len(item_nodes)
    g["category"].node_id = torch.tensor(cat_nodes, dtype=torch.long)
    g["category"].num_nodes = len(cat_nodes)
    if has_taxonomy:
        g["parent"].node_id = torch.tensor(parent_nodes, dtype=torch.long)
        g["parent"].num_nodes = len(parent_nodes)

    # --- Edge: sequential (theo thứ tự click NGUYÊN THỦY của phiên) ---
    seq_src = [item_local[session[j]] for j in range(len(session) - 1)]
    seq_dst = [item_local[session[j + 1]] for j in range(len(session) - 1)]
    seq = torch.tensor([seq_src, seq_dst], dtype=torch.long).reshape(2, -1)

    # --- Edge: membership (mỗi unique item → category của nó) ---
    mem_src = [item_local[it] for it in item_nodes]
    mem_dst = [cat_local[item2cat[it]] for it in item_nodes]
    mem = torch.tensor([mem_src, mem_dst], dtype=torch.long).reshape(2, -1)

    g["item", "sequential", "item"].edge_index = seq
    g["item", "rev_sequential", "item"].edge_index = seq.flip(0)
    g["item", "membership", "category"].edge_index = mem
    g["category", "rev_membership", "item"].edge_index = mem.flip(0)

    # --- Edge: taxonomy (leaf category → parent, chỉ khi có taxonomy) ---
    if has_taxonomy:
        tax_src, tax_dst = [], []
        for c in cat_nodes:
            p = cat_parent.get(c)
            if p is not None:
                tax_src.append(cat_local[c])
                tax_dst.append(parent_local[p])
        tax = torch.tensor([tax_src, tax_dst], dtype=torch.long).reshape(2, -1)
        g["category", "taxonomy", "parent"].edge_index = tax
        g["parent", "rev_taxonomy", "category"].edge_index = tax.flip(0)

    # Vị trí local của item CUỐI phiên — cần cho soft-attention readout
    g["item"].last_idx = torch.tensor([item_local[session[-1]]], dtype=torch.long)

    # --- Node/Edge: star node ảo (tùy chọn, SGNN-HN) ---
    if add_star_node:
        n_local = len(item_nodes)
        g["session"].num_nodes = 1
        star_src = list(range(n_local))
        star_dst = [0] * n_local
        star = torch.tensor([star_src, star_dst], dtype=torch.long).reshape(2, -1)
        g["item", "to_star", "session"].edge_index = star
        g["session", "from_star", "item"].edge_index = star.flip(0)

    return g


def sessions_to_batch(
    sessions: list[list[int]],
    item2cat: dict[int, int],
    cat_parent: dict[int, int] | None,
    use_taxonomy: bool,
    add_star_node: bool = False,
) -> Batch:
    """Xây graph cho từng phiên rồi gom thành một PyG Batch."""
    graphs = [
        session_to_graph(s, item2cat, cat_parent, use_taxonomy, add_star_node)
        for s in sessions
    ]
    return Batch.from_data_list(graphs)
