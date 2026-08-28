from extractor import extract_pdf
from chunker import chunk_pages


pdf_path = "data/lectures/MTH441/lecture01.pdf"


pages = extract_pdf(
    pdf_path,
    course="MTH441",
    lecture="Lecture 01"
)


chunks = chunk_pages(pages)


print("Number of pages :", len(pages))
print("Number of chunks:", len(chunks))
jhol = 0
jhol2 = 0
noice = 0

for i, chunk in enumerate(chunks):

    if 1:

        print("\n" + "=" * 70)
        print(f"CHUNK {i + 1}")
        print("=" * 70)

        print(chunk["metadata"])
        print()
        print(chunk["text"])
        if(len(chunk["text"])>800):
            #print("JHOL")
            jhol+=1
        if(i>0):
            if(chunks[i]["text"][0:120]==chunks[i-1]["text"][-120:]):
                #print("JHOL2")
                noice+=1
            else:
                jhol2+=1
                numzz=0
                while(numzz<120):
                    if(chunks[i]["text"][numzz]!=chunks[i-1]["text"][-120+numzz]):
                        print(f"Mismatch at position {numzz}: {chunks[i]['text'][numzz]} != {chunks[i-1]['text'][-120+numzz]}")
                    numzz+=1


print(f"Number of chunks with more than 800 characters: {jhol}")
print(f"Number of chunks with not proper overlapping text: {jhol2}")