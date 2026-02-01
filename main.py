import ollama
from unstructured.partition.pdf import partition_pdf
import base64
import chromadb
import os


folder_path = "./content/"


def main():

    prompt = str(input(">>> "))
    collection = initialize()
    chat(prompt,collection)

def initialize():
    client = chromadb.PersistentClient('./chromadb')
    collection = client.get_or_create_collection(name="docs")

    chunk_id  = 0;

    if collection.count() == 0:

        for file in os.scandir(folder_path):
            if file.is_file():
                print(f"Processing {file.name}")
                file_path = str(file.path)

                chunks = partition_pdf(
                    filename=file_path,
                    infer_table_structure=True,
                    strategy="hi_res",
                    extract_image_block_types=["Image"],
                    #image_output_dir_path=output_path,   # if None, images and tables will saved in base64

                    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

                    chunking_strategy="by_title",          # or 'basic'
                    max_characters=2000,                  # defaults to 500
                    combine_text_under_n_chars=500,       # defaults to 0
                    new_after_n_chars=1500,

                )


                for chunk in chunks:

                    if "Table" in str(type(chunk)):
                        content = chunk.metadata.text_as_html
                        content_type = "table"
                    else:
                        content = str(chunk).strip()
                        content_type = "text"

                    if not content:
                        continue


                    response = ollama.embed(model = "nomic-embed-text", input = content)
                    embedding = response["embeddings"][0]

                    collection.add(
                        ids=[f"{file.name}_{chunk_id}"],
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[{
                            "source": file.name,
                            "type": content_type
                        }]
                    )
                    chunk_id += 1
                print(f"Successfully indexed {chunk_id} items.")
                images = get_images_base64(chunks)
                return collection

    else:
        print(f"Database already contains {collection.count()} chunks. Skipping indexing.")
        return collection


def chat(prompt,collection):

    response = ollama.embed(
        model="nomic-embed-text",
        input=prompt
    )
    results = collection.query(
        query_embeddings=response["embeddings"],
        n_results = 5
    )
    all_context = "\n\n".join(results['documents'][0])
    #print(f"--- RETRIEVED CONTEXT ---\n{all_context}\n------------------------")

    output = ollama.generate(
        model = "llava:7b",
        prompt = f"Answer ONLY using this context: {all_context}. Question: {prompt}"
    )
    print(output['response'])

def get_images_base64(chunks):

    images_b64 = []
    for chunk in chunks:

        if "CompositeElement" in str(type(chunk)):

            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):

                    images_b64.append(el.metadata.image_base64)
    return images_b64

def display_base64_image(base64_code,index):
    image_data = base64.b64decode(base64_code)
    with open(f"image({index}).png", "wb") as f:
              f.write(image_data)

if __name__ == "__main__":

    main()

