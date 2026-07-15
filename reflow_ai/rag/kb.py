"""知识库 RAG（本地，数据不出厂）。

对应 V2 §8。演示用 TF-IDF 检索 + 规则拼接；
生产替换为 BGE-M3 Embedding + pgvector 检索 + 本地 Qwen-7B 生成。
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import CONFIG


class KBDocStore:
    def __init__(self):
        self.docs = []
        self.vec = TfidfVectorizer()
        self._fit = False

    def add(self, content, source, category):
        self.docs.append({"content": content, "source": source, "category": category})

    def build(self):
        if self.docs:
            self.mat = self.vec.fit_transform([d["content"] for d in self.docs])
            self._fit = True

    def search(self, question, top_k=5):
        if not self._fit:
            return []
        q = self.vec.transform([question])
        sim = cosine_similarity(q, self.mat)[0]
        idx = np.argsort(sim)[::-1][:top_k]
        return [self.docs[i] for i in idx if sim[i] > 0]

    def answer(self, question):
        hits = self.search(question, 5)
        context = "\n---\n".join(d["content"] for d in hits)
        sources = [d["source"] for d in hits]
        # 生产：送入本地 Qwen-7B 生成（带引用）；此处返回规则拼接
        reply = f"[基于知识库 {len(hits)} 条相关经验] " + context[:500]
        return {"answer": reply, "sources": sources}


def build_default_kb():
    """预置示例知识；生产从 SOP / 老师傅经验 / 历史案例批量导入。"""
    kb = KBDocStore()
    kb.add("BGA 区域虚焊多因峰值温度不足或 TAL 过短，建议回流峰值+6~8℃、链速-5cm/min。",
           "经验#E017", "虚焊")
    kb.add("桥连常因峰值过高或链速过慢导致过量塌陷，建议回流峰值-3~5℃并检查钢网开窗。",
           "经验#E023", "桥连")
    kb.add("立碑多因两端受热不均或焊盘不对称，建议减小升温斜率并检查元件两端焊盘尺寸。",
           "经验#E031", "立碑")
    kb.add("锡珠常因升温过快助焊剂飞溅或湿度超标，建议预热150~170℃延长停留，来料湿度≤40%RH。",
           "经验#E045", "锡珠")
    kb.build()
    return kb
