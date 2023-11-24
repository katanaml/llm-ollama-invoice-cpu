import timeit
import argparse
from rag.pipeline import build_rag_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What is the invoice number value?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    start = timeit.default_timer()

    retriever = build_rag_pipeline()
    answer = retriever.get_relevant_documents(args.input)

    end = timeit.default_timer()

    print(f'\nAnswer:\n {answer}')
    print('=' * 50)

    print(f"Time to retrieve answer: {end - start}")