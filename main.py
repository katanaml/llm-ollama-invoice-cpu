import timeit
import argparse
from rag.pipeline import build_rag_pipeline
import json


def get_rag_response(query, chain):
    response = chain({'query': query})

    res = response['result']

    start_index = res.find('{')
    end_index = res.rfind('}')

    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_fragment = res[start_index:end_index + 1]
        try:
            # Convert the extracted string to JSON
            json_data = json.loads(json_fragment)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    else:
        print("No JSON object found in the string.")

    return res


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