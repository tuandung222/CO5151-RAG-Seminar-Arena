import json
from typing import List, Dict, Any, Tuple
import networkx as nx
from .retriever import LegalRetriever
from .llm import UnifiedLLM


class GraphRAGPipeline:
    """
    GraphRAG Pipeline (Edge et al., Microsoft Research 2024):
    1. Knowledge Graph Representation (Nodes = Articles/Entities, Edges = Cross-references/Thematic links)
    2. Hierarchical Community Detection (Leiden / Modularity Clusters)
    3. Community-level Map-Reduce Summarization for Global Synthesis Questions
    """

    def __init__(self, retriever: LegalRetriever, llm: UnifiedLLM):
        self.retriever = retriever
        self.llm = llm
        self.graph = nx.Graph()
        self._build_legal_graph()

    def _build_legal_graph(self):
        """Construct the legal knowledge graph from the corpus."""
        # Add nodes with metadata
        for doc in self.retriever.corpus:
            self.graph.add_node(
                doc["article_id"],
                title=doc.get("title", ""),
                content=doc.get("content", ""),
            )

        # Cross-reference & thematic edges
        edges = [
            ("Dieu_13", "Dieu_20", {"relation": "quy_dinh_loai_hop_dong"}),
            ("Dieu_13", "Dieu_24", {"relation": "thoa_thuan_thu_viec"}),
            ("Dieu_24", "Dieu_25", {"relation": "quy_dinh_thoi_gian_thu_viec"}),
            ("Dieu_24", "Dieu_26", {"relation": "quy_dinh_tien_luong_thu_viec"}),
            ("Dieu_24", "Dieu_27", {"relation": "ket_thuc_thu_viec"}),
            ("Dieu_20", "Dieu_34", {"relation": "het_han_va_cham_dut"}),
            ("Dieu_34", "Dieu_35", {"relation": "nld_don_phuong"}),
            ("Dieu_34", "Dieu_36", {"relation": "nsdld_don_phuong"}),
            ("Dieu_36", "Dieu_37", {"relation": "han_che_quyen_don_phuong"}),
            ("Dieu_35", "Dieu_40", {"relation": "don_phuong_trai_phap_luat"}),
            ("Dieu_36", "Dieu_41", {"relation": "boi_thuong_trai_phap_luat"}),
            ("Dieu_34", "Dieu_46", {"relation": "huong_tro_cap_thoi_viec"}),
            ("Dieu_37", "Dieu_122", {"relation": "bao_ve_lao_dong_nu_va_om_dau"}),
            ("Dieu_122", "Dieu_125", {"relation": "hinh_thuc_sa_thai"}),
            ("Dieu_36", "Dieu_125", {"relation": "hanh_vi_tu_y_bo_viec"}),
        ]
        for u, v, data in edges:
            if self.graph.has_node(u) and self.graph.has_node(v):
                self.graph.add_edge(u, v, **data)

    def get_communities(self) -> List[Dict[str, Any]]:
        """Run community detection on the legal graph using Greedy Modularity."""
        communities_raw = list(nx.community.greedy_modularity_communities(self.graph))
        results = []

        thematic_names = [
            "Cụm 1: Giao kết & Chế định Thử việc (Hợp đồng, Thời hạn, Lương thử việc)",
            "Cụm 2: Chấm dứt HĐLĐ & Quyền đơn phương, Bồi thường",
            "Cụm 3: Kỷ luật lao động, Sa thải & Bảo vệ lao động đặc thù",
        ]

        for i, comm in enumerate(communities_raw):
            node_ids = sorted(list(comm))
            articles = [self.retriever.get_article(nid) for nid in node_ids]
            results.append({
                "community_id": i + 1,
                "name": thematic_names[i] if i < len(thematic_names) else f"Cộng đồng {i+1}",
                "node_count": len(node_ids),
                "article_ids": node_ids,
                "articles": articles,
            })
        return results

    def run_graph_rag(self, query: str) -> Dict[str, Any]:
        """
        Execute GraphRAG Community Summary Map-Reduce workflow:
        1. MAP: Generate partial answers for each community cluster.
        2. REDUCE: Synthesize partial community answers into a comprehensive global answer.
        """
        communities = self.get_communities()
        map_summaries = []

        # MAP Phase
        for comm in communities:
            comm_docs_text = ""
            for a in comm["articles"]:
                comm_docs_text += f"- {a.get('title')}: {a.get('content')}\n"

            map_prompt = f"""Dưới đây là các điều luật thuộc cụm chuyên đề: "{comm['name']}".
Hãy trích xuất và tóm tắt những quy định trong cụm này có liên quan đến câu hỏi toàn cục: "{query}".
Nếu cụm này không chứa thông tin trực tiếp, hãy ghi nhận các nguyên tắc chung.

ĐIỀU LUẬT TRONG CỤM:
{comm_docs_text}

TÓM TẮT ĐÓNG GÓP TỪ CỤM NÀY:"""

            res = self.llm.generate(map_prompt, max_tokens=300)
            map_summaries.append({
                "community_id": comm["community_id"],
                "community_name": comm["name"],
                "article_ids": comm["article_ids"],
                "summary": res["text"],
                "latency_ms": res.get("latency_ms", 0),
            })

        # REDUCE Phase
        combined_summaries_text = ""
        for item in map_summaries:
            combined_summaries_text += f"=== {item['community_name']} ===\n{item['summary']}\n\n"

        reduce_prompt = f"""Bạn là một chuyên gia pháp lý. Hãy tổng hợp toàn diện câu trả lời cho câu hỏi sau dựa trên các báo cáo tóm tắt từ từng cụm chuyên đề pháp lý:

CÂU HỎI TOÀN CỤC:
{query}

BÁO CÁO TỪNG CỤM CHUYÊN ĐỀ (MAP OUTPUT):
{combined_summaries_text}

HÃY ĐƯA RA CÂU TRẢ LỜI TỔNG THỂ TOÀN DIỆN (REDUCE SYNTHESIS), NÊU RÕ CÁC NHÓM TRƯỜNG HỢP VÀ ĐIỀU LUẬT ÁP DỤNG:"""

        reduce_res = self.llm.generate(reduce_prompt, max_tokens=600)

        return {
            "mode": "GraphRAG (Hierarchical Community Map-Reduce)",
            "query": query,
            "communities": communities,
            "map_summaries": map_summaries,
            "global_answer": reduce_res["text"],
            "total_latency_ms": round(sum(m.get("latency_ms", 0) for m in map_summaries) + reduce_res.get("latency_ms", 0), 2),
            "model": reduce_res["model"],
        }
