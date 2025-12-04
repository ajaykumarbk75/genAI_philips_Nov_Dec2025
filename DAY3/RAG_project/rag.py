# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable tokenizer parallelism to avoid warnings

# import torch # PyTorch for device checks and potential model tensors
# from sentence_transformers import SentenceTransformer # embedding model wrapper
# import faiss # FAISS for vector index storage and search
# from transformers import AutoTokenizer, AutoModelForCausalLM # HF model/tokenizer is used for LLM

# # ---------------------------------------------------   
# # DEVICE SETUP (IMPORTANT FOR MAC)
# # ---------------------------------------------------
# device = "mps" if torch.backends.mps.is_available() else "cpu" # prefer MPS on Mac, fallback to CPU
# print("Using device:", device)

# ---------------------------------------------------
import os  # operating system interfaces
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer parallelism to avoid warnings

import json  # JSON read/write for metadata persistence
from pathlib import Path  # filesystem path utilities
import argparse  # command-line argument parsing

import torch  # PyTorch for device checks and potential model tensors
from sentence_transformers import SentenceTransformer  # embedding model wrapper
import faiss  # FAISS for vector index storage and search
from transformers import AutoTokenizer, AutoModelForCausalLM  # HF model/tokenizer

# ---------------------------------------------------
# DEVICE SETUP (IMPORTANT FOR MAC)
# ---------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"  # prefer MPS on Mac, fallback to CPU
print("Using device:", device)  # show chosen device

# ---------------------------------------------------
# Helper: documents directory and persistence folder
# ---------------------------------------------------
ROOT = Path(__file__).resolve().parent  # directory containing this script
DOCS_DIR = ROOT / "documents"  # expected documents folder next to this file
PERSIST_DIR = ROOT / "faiss_store"  # folder where index + metadata will be saved
PERSIST_DIR.mkdir(parents=True, exist_ok=True)  # create persistence folder if missing


def load_documents():
    docs_list = []  # accumulator for document strings
    if not DOCS_DIR.exists():
        return docs_list  # return empty when no documents folder
    for file in sorted(DOCS_DIR.iterdir()):
        if file.is_file():
            with open(file, "r", encoding="utf-8") as fh: # open file for reading
                docs_list.append(fh.read())  # append file contents
    return docs_list  # return list of document texts


# ---------------------------------------------------
# 2. Embeddings + Vector DB (with persistence helpers)
# ---------------------------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # sentence-transformers model

# Globals to be initialized by building or loading the index
docs = []  # list of document strings
index = None  # FAISS index object reference


def build_index_from_docs(document_texts): # build FAISS index from list of document texts
    global docs, index  # modify module-level variables
    docs = list(document_texts)  # copy incoming documents
    if len(docs) == 0:
        index = None  # nothing to index
        return
    embeddings = embedder.encode(docs, convert_to_numpy=True)  # compute embeddings as numpy array
    embeddings = embeddings.astype('float32')  # FAISS expects float32
    dim = embeddings.shape[1]  # embedding dimensionality
    index = faiss.IndexFlatL2(dim)  # L2 (Euclidean) index means exact search - good for small datasets
    index.add(embeddings)  # add vectors to index


def save_index(save_dir: Path = PERSIST_DIR):
    """Save FAISS index and metadata (docs) to disk."""
    global index, docs
    save_dir = Path(save_dir)  # ensure Path object
    save_dir.mkdir(parents=True, exist_ok=True)  # create directory if needed
    idx_path = save_dir / "faiss.index"  # index filename
    meta_path = save_dir / "metadata.json"  # metadata filename
    if index is None:
        raise RuntimeError("No index to save")  # nothing to persist
    faiss.write_index(index, str(idx_path))  # write FAISS index to file
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"docs": docs}, fh, ensure_ascii=False, indent=2)  # write docs metadata
    print(f"Saved index -> {idx_path}")  # confirm save path


def load_index(load_dir: Path = PERSIST_DIR):
    """Load FAISS index and metadata from disk into globals."""
    global index, docs
    load_dir = Path(load_dir)  # ensure Path
    idx_path = load_dir / "faiss.index"  # expected index file
    meta_path = load_dir / "metadata.json"  # expected metadata file
    if not idx_path.exists() or not meta_path.exists():
        print("No persisted index found.")  # nothing to load
        return False
    loaded_index = faiss.read_index(str(idx_path))  # read FAISS index
    with open(meta_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)  # load metadata
    docs = payload.get("docs", [])  # restore docs list
    index = loaded_index  # set global index
    print(f"Loaded index with {len(docs)} documents from {load_dir}")  # show count
    return True  # success


# ---------------------------------------------------
# 3. Load a Mac-friendly LLM (optional)
# ---------------------------------------------------
model_name = "gpt2"  # "tiiuae/falcon-rw-1b" - default small-ish model choice - CAN USE gpt2 also 1
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # load tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)  # load model weights
    model.to(device)  # move model to chosen device
    llm_available = True  # mark that generation is possible
except Exception as exc:
    print("Could not load LLM; continuing without generation:", exc)  # warn and continue
    tokenizer = None  # no tokenizer available
    model = None  # no model available
    llm_available = False  # generation disabled


# ---------------------------------------------------
# 4. RAG Function
# ---------------------------------------------------
def ask(question, top_k: int = 1): # retrieve + generate answer to question using RAG
    global index, docs
    if index is None:
        return "No index available. Ingest documents first."  # guard when no index
    q_emb = embedder.encode([question], convert_to_numpy=True).astype('float32')  # embed query
    D, I = index.search(q_emb, k=min(top_k, len(docs)))  # search index for nearest neighbors, D=distances, I=indices,
    #k=min - means we don't request more neighbors than we have documents
    #top_k - number of nearest neighbors to retrieve
    # I is shape (1, k) since we queried with a single vector
    retrieved = []  # list to hold retrieved document texts
    for idx in I[0]:
        if idx < 0 or idx >= len(docs):
            continue  # skip invalid indices
        retrieved.append(docs[idx])  # append matching document text

    if llm_available:
        context = "\n\n".join(retrieved)  # join contexts with spacing
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"  # craft prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # tokenize and move to device
        #output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.3)  # generate
        output = model.generate( **inputs,max_new_tokens=150, do_sample=True,temperature=0.3,repetition_penalty=1.2)
        #max tokens 150 - limit response length
        #temperature 0.3 - low temperature for more focused answers - means less randomness
        #temperature means higher values = more randomness, lower values = more focused
        ans = tokenizer.decode(output[0], skip_special_tokens=True)  # decode generated tokens
        return ans  # return model answer
    else:
        return "Retrieved context:\n" + "\n\n".join(retrieved) + "\n\nBased on the above, here is a short answer to your question:\n[answer not generated: no LLM configured]"  # fallback


# ---------------------------------------------------
# 5. CLI / startup behavior
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simple RAG demo with FAISS persistence")  # CLI parser
    parser.add_argument("--save", action="store_true", help="Save built index to disk")  # save flag
    parser.add_argument("--load", action="store_true", help="Load index from disk if present")  # load flag
    args = parser.parse_args()  # parse args

    if args.load:
        loaded = load_index()  # attempt to load persisted index
        if loaded:
            print("Index loaded; ready.")  # loaded successfully
        else:
            print("No persisted index; building from documents...")  # fallback to building
            build_index_from_docs(load_documents())  # build index from files
    else:
        build_index_from_docs(load_documents())  # default behavior: build from documents

    if args.save:
        if index is None:
            print("Nothing to save (empty index)")  # nothing to persist
        else:
            save_index()  # save current index and docs

    print("RAG demo ready!")  # ready prompt for interactive loop
    try:
        while True:
            q = input("\nAsk: ")  # prompt user
            if not q:
                continue  # ignore empty lines
            print("\nAnswer:\n")  # header for the answer
            print(ask(q))  # run query and print result
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")  # graceful exit on Ctrl+C or EOF


if __name__ == "__main__":  # here to run when executed as script - it is constructror 
    main()  # entrypoint when executed as a script
