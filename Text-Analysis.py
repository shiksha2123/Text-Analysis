from dotenv import load_dotenv
import os

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

def main():
    try:
        # Load environment variables
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Create the Text Analytics client
        credential = AzureKeyCredential(ai_key)
        ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

        # Analyze each text file in the 'reviews' folder
        reviews_folder = 'reviews'
        for file_name in os.listdir(reviews_folder):
            print('\n-------------\n' + file_name)

            # Read file contents
            file_path = os.path.join(reviews_folder, file_name)
            text = open(file_path, encoding='utf8').read()
            print('\n' + text)

            # Detect language
            language_result = ai_client.detect_language(documents=[text])[0]
            print('\nLanguage: {}'.format(language_result.primary_language.name))

            # Analyze sentiment
            sentiment_result = ai_client.analyze_sentiment(documents=[text])[0]
            print("\nSentiment: {}".format(sentiment_result.sentiment))

            # Extract key phrases
            key_phrases_result = ai_client.extract_key_phrases(documents=[text])[0]
            if key_phrases_result.key_phrases:
                print("\nKey Phrases:")
                for phrase in key_phrases_result.key_phrases:
                    print('\t{}'.format(phrase))

            # Recognize entities
            entities_result = ai_client.recognize_entities(documents=[text])[0]
            if entities_result.entities:
                print("\nEntities:")
                for entity in entities_result.entities:
                    print('\t{} ({})'.format(entity.text, entity.category))

            # Recognize linked entities
            linked_entities_result = ai_client.recognize_linked_entities(documents=[text])[0]
            if linked_entities_result.entities:
                print("\nLinked Entities:")
                for linked_entity in linked_entities_result.entities:
                    print('\t{} ({})'.format(linked_entity.name, linked_entity.url))

    except Exception as ex:
        print("Error:", ex)

if __name__ == "__main__":
    main()
