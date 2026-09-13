from src.vectorstore import get_collection
import numpy as np

col = get_collection(reset=True)          # สร้างคลังเปล่า
col.add(ids=["a","b"], embeddings=np.random.rand(2,384).tolist(),
        documents=["hello","world"], metadatas=[{"rating":1},{"rating":5}])
print(col.count())                         # → 2
col2 = get_collection()                    # ไม่ reset — ควรเจอของเดิม
print(col2.count())                        # → ยัง 2 = persistent จริง
col3 = get_collection(reset=True)          # reset — ควรว่าง
print(col3.count())                        # → 0 = idempotent จริง