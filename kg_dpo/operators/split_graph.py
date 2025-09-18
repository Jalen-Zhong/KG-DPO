import random
from collections import defaultdict
from tqdm.asyncio import tqdm as tqdm_async
from kg_dpo.utils import logger
from kg_dpo.models import NetworkXStorage
from kg_dpo.configs import KGWalkingConfig

async def _get_node_info(
    node_id: str,
    graph_storage: NetworkXStorage,
)-> dict:
    """
    Get node info

    :param node_id: node id
    :param graph_storage: graph storage instance
    :return: node info
    """
    node_data = await graph_storage.get_node(node_id)
    return {
        "node_id": node_id,
        **node_data
    }


def _get_level_n_edges_by_max_tokens(
        edge_adj_list: dict,
        node_dict: dict,
        edges: list,
        nodes: list,
        src_edge: tuple,
        max_depth: int,
        bidirectional: bool,
        max_tokens: int,
) -> list:
    """
    Get level n edges for an edge.
    n is decided by max_depth in traverse_strategy.

    :param edge_adj_list
    :param node_dict
    :param edges
    :param nodes
    :param src_edge
    :param max_depth
    :param bidirectional
    :param max_tokens
    :return: level n edges
    """
    src_id, tgt_id, src_edge_data = src_edge

    max_tokens -= (src_edge_data["length"] + nodes[node_dict[src_id]][1]["length"]
                   + nodes[node_dict[tgt_id]][1]["length"])

    level_n_edges = []

    start_nodes = {tgt_id} if not bidirectional else {src_id, tgt_id}
    temp_nodes = {src_id, tgt_id}

    while max_depth > 0 and max_tokens > 0:
        max_depth -= 1

        candidate_edges = [
            edges[edge_id]
            for node in start_nodes
            for edge_id in edge_adj_list[node]
            if not edges[edge_id][2].get("visited", False)
        ]

        if not candidate_edges:
            break

        for edge in candidate_edges:
            max_tokens -= edge[2]["length"]
            if not edge[0] in temp_nodes:
                max_tokens -= nodes[node_dict[edge[0]]][1]["length"]
            if not edge[1] in temp_nodes:
                max_tokens -= nodes[node_dict[edge[1]]][1]["length"]

            if max_tokens < 0:
                return level_n_edges

            level_n_edges.append(edge)
            edge[2]["visited"] = True
            temp_nodes.add(edge[0])
            temp_nodes.add(edge[1])

        new_start_nodes = set()
        for edge in candidate_edges:
            if not edge[0] in start_nodes:
                new_start_nodes.add(edge[0])
            if not edge[1] in start_nodes:
                new_start_nodes.add(edge[1])

        start_nodes = new_start_nodes

    return level_n_edges

async def get_batches_with_strategy( # pylint: disable=too-many-branches
    nodes: list,
    edges: list,
    graph_storage: NetworkXStorage,
    walking_strategy: KGWalkingConfig
):
    max_depth = walking_strategy.MAX_DEPTH

    # 构建临接矩阵
    edge_adj_list = defaultdict(list)
    node_dict = {}
    processing_batches = []

    node_cache = {}

    async def get_cached_node_info(node_id: str) -> dict:
        if node_id not in node_cache:
            node_cache[node_id] = await _get_node_info(node_id, graph_storage)
        return node_cache[node_id]

    for i, (node_name, _) in enumerate(nodes):
        node_dict[node_name] = i

    for i, (src, tgt, _) in enumerate(edges):
        edge_adj_list[src].append(i)
        edge_adj_list[tgt].append(i)

    for edge in tqdm_async(edges, desc="Preparing batches"):
        if "visited" in edge[2] and edge[2]["visited"]:
            continue

        edge[2]["visited"] = True

        _process_nodes = []
        _process_edges = []

        src_id = edge[0]
        tgt_id = edge[1]

        _process_nodes.extend([await get_cached_node_info(src_id),
                               await get_cached_node_info(tgt_id)])
        _process_edges.append(edge)

        level_n_edges = _get_level_n_edges_by_max_tokens(
            edge_adj_list, node_dict, edges, nodes, edge, max_depth,
            walking_strategy.BIDRECTIONAL, walking_strategy.MAX_TOKENS
        )

        for _edge in level_n_edges:
            _process_nodes.append(await get_cached_node_info(_edge[0]))
            _process_nodes.append(await get_cached_node_info(_edge[1]))
            _process_edges.append(_edge)

        # 去重
        _process_nodes = list({node['node_id']: node for node in _process_nodes}.values())
        _process_edges = list({(edge[0], edge[1]): edge for edge in _process_edges}.values())

        processing_batches.append((_process_nodes, _process_edges))

    logger.info("Processing batches: %d", len(processing_batches))

    # isolate nodes
    isolated_node_strategy = walking_strategy.ISOLATED_NODE_STRATEGY
    if isolated_node_strategy == "add":
        processing_batches = await _add_isolated_nodes(nodes, processing_batches, graph_storage)
        logger.info("Processing batches after adding isolated nodes: %d", len(processing_batches))

    return processing_batches

async def _add_isolated_nodes(
        nodes: list,
        processing_batches: list,
        graph_storage: NetworkXStorage,
) -> list:
    visited_nodes = set()
    for _process_nodes, _process_edges in processing_batches:
        for node in _process_nodes:
            visited_nodes.add(node["node_id"])
    for node in nodes:
        if node[0] not in visited_nodes:
            _process_nodes = [await _get_node_info(node[0], graph_storage)]
            processing_batches.append((_process_nodes, []))
    return processing_batches
