import os
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


def load_json_data(file_path):
    """JSON 파일에서 데이터를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_texts(json_data):
    """
    JSON의 "list" 항목에서 각 항목의 articleTitle과 articleText를 결합하여
    하나의 문서 문자열로 반환합니다.
    """
    texts = []
    for item in json_data.get("list", []):
        title = item.get("articleTitle", "")
        text = item.get("articleText", "")
        full_text = title + "\n" + text
        texts.append(full_text)
    return texts


def chunk_text(text, chunk_size=500, overlap=50):
    """
    주어진 텍스트를 지정된 길이(chunk_size)로 겹침(overlap)과 함께 청크로 분할합니다.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


def prepare_chunks(texts, chunk_size=500, overlap=50):
    """
    여러 문서 텍스트를 받아서 모든 청크를 리스트로 만들어 반환합니다.
    """
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks


def build_faiss_index(embeddings):
    """
    생성된 임베딩(NumPy 배열)을 바탕으로 FAISS의 L2 인덱스를 구축합니다.
    """
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.astype(np.float32))
    return index


def save_preprocessed_data(chunks, embeddings, index, output_dir="preprocessing"):
    """
    전처리된 청크, 임베딩, FAISS 인덱스를 지정된 디렉토리에 저장합니다.

    저장 파일:
    - preprocessed_chunks.json : 청크 리스트 저장
    - embeddings.npy           : 임베딩 NumPy 배열 저장
    - faiss_index.index        : FAISS 인덱스 저장
    - preprocessed_data.pkl    : 전체 데이터를 하나의 pickle 파일로 저장
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 청크 저장 (JSON 파일)
    chunks_path = os.path.join(output_dir, "preprocessed_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 2. 임베딩 저장 (NumPy 파일)
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)

    # 3. FAISS 인덱스 저장
    faiss_index_path = os.path.join(output_dir, "faiss_index.index")
    faiss.write_index(index, faiss_index_path)

    # 4. 전체 전처리 데이터를 pickle 파일로 저장
    pickle_path = os.path.join(output_dir, "preprocessed_data.pkl")
    data_to_save = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(data_to_save, f)


def main():
    # JSON 파일 경로 (첨부된 파일)
    input_file = "hagsailjeong.json"
    json_data = load_json_data(input_file)

    # 1. JSON 데이터에서 문서(기사 제목+본문) 추출
    texts = extract_texts(json_data)
    print(f"총 {len(texts)}개의 문서를 추출했습니다.")

    # 2. 문서를 청크로 분할 (청크 크기 500자, 겹침 50자)
    all_chunks = prepare_chunks(texts, chunk_size=500, overlap=50)
    print(f"전처리된 총 {len(all_chunks)}개의 청크가 생성되었습니다.")

    # 3. 임베딩 모델 로드 (SentenceTransformer)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. 각 청크를 벡터 임베딩으로 변환
    embeddings = model.encode(all_chunks)
    embeddings = np.array(embeddings)

    # 5. FAISS 인덱스 구축
    index = build_faiss_index(embeddings)

    # 6. 전처리 데이터를 "preprocessing" 디렉토리에 저장
    save_preprocessed_data(all_chunks, embeddings, index, output_dir="preprocessing")
    print("전처리 데이터가 'preprocessing' 디렉토리에 저장되었습니다.")


if __name__ == "__main__":
    main()
