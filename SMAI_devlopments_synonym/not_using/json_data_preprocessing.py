import os
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


# JSON 파일에서 데이터를 로드하는 함수
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# JSON 데이터에서 기사 제목과 본문을 추출하여 하나의 텍스트로 결합하는 함수
def extract_texts(json_data):
    texts = []
    for item in json_data.get("list", []):
        title = item.get("articleTitle", "")
        content = item.get("articleText", "")
        full_text = title + "\n" + content
        texts.append(full_text)
    return texts


# 텍스트를 지정한 크기(chunk_size)와 겹침(overlap)을 적용해 청크로 분할하는 함수
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# 여러 텍스트에 대해 청크를 생성하는 함수
def prepare_chunks(texts, chunk_size=500, overlap=50):
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks


# FAISS 인덱스 생성 함수
def build_faiss_index(embeddings):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))
    return index


# 쿼리 문장에 대해 FAISS 인덱스에서 top_k와 유사한 청크를 검색하는 함수
def search_query(query, model, index, all_chunks, top_k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), top_k)
    results = [all_chunks[i] for i in indices[0]]
    return results


# 전처리 데이터(청크, 임베딩, 인덱스)를 'preprocessing' 디렉토리에 저장하는 함수
def save_preprocessed_data(all_chunks, embeddings, index):
    output_dir = "preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    # 청크 저장 (JSON 파일)
    chunks_path = os.path.join(output_dir, "preprocessed_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # 임베딩 저장 (NumPy 파일)
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)

    # FAISS 인덱스 저장
    faiss_index_path = os.path.join(output_dir, "faiss_index.index")
    faiss.write_index(index, faiss_index_path)

    # 전체 데이터를 하나의 pickle 파일로 저장 (옵션)
    preprocessed_data = {"chunks": all_chunks, "embeddings": embeddings}
    pickle_path = os.path.join(output_dir, "preprocessed_data.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(preprocessed_data, f)


if __name__ == '__main__':
    # 1. JSON 파일에서 데이터 로드 (파일명: hagsailjeong.json)
    file_path = "hagsailjeong.json"
    json_data = load_json_data(file_path)

    # 2. 기사 제목과 본문을 결합해 텍스트 리스트 생성
    texts = extract_texts(json_data)

    # 3. 텍스트를 청크로 분할 (청크 크기 500자, 겹침 50자)
    all_chunks = prepare_chunks(texts, chunk_size=500, overlap=50)
    print(f"전체 {len(all_chunks)}개의 청크 생성됨.")

    # 4. SentenceTransformer 모델 로드 및 각 청크의 임베딩 생성 (예: all-MiniLM-L6-v2)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_chunks)

    # 5. FAISS 인덱스 생성
    index = build_faiss_index(np.array(embeddings))

    # 6. 전처리 데이터를 'preprocessing' 디렉토리에 저장
    save_preprocessed_data(all_chunks, embeddings, index)

    # 7. 예제 쿼리 수행 및 결과 출력 (예: "2024-1학기 수강신청")
    query = "2024-1학기 수강신청"
    results = search_query(query, model, index, all_chunks, top_k=3)
    print("==== 검색 질의:", query, "====")
    for i, res in enumerate(results, start=1):
        print(f"\n-- Top {i} 청크 --")
        print(res)
