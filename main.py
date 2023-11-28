import timeit
import argparse
from rag.pipeline import build_rag_pipeline

def get_rag_response(query, chain):
    response = chain({'query': query})

    return response['result']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What is the invoice number value?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    start = timeit.default_timer()

    qa_chain = build_rag_pipeline()
    print('Retrieving answer...')
    answer = get_rag_response(args.input, qa_chain)

    end = timeit.default_timer()

    print(f'\nAnswer:\n {answer}')
    print('=' * 50)

    print(f"Time to retrieve answer: {end - start}")