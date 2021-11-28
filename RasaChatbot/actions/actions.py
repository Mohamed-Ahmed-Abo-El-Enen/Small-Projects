# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
import requests
import json


def check_exist_country(country):
    api_link = "https://qcooc59re3.execute-api.us-east-1.amazonaws.com/dev/getCountries"
    json_data = requests.get(api_link).json()
    if country.lower() in list(map(str.lower, json_data["body"])):
        return True
    return False

class ActionCheckCapital(Action):
    __api_link = "https://qcooc59re3.execute-api.us-east-1.amazonaws.com/dev/getCapital"
    
    def name(self)-> Text:
        return "action_get_capital"
    
    def run(self, dispatcher, tracker, domain) -> List[Dict[Text, Any]]:
        country = tracker.get_slot('country')
        if not check_exist_country(country):
            dispatcher.utter_message("No Country with this name in our dataset")
            return []
		
        json_data = requests.post(self.__api_link , json.dumps({'country': country})).json()
        capital = json_data["body"]["capital"]
        response = "capital is {}".format(capital)
        dispatcher.utter_message(response)
        return []
    	
    
    
class ActionCheckPopulation(Action):
    __api_link = "https://qcooc59re3.execute-api.us-east-1.amazonaws.com/dev/getPopulation"

    def name(self)-> Text:
        return "action_get_population"
    
    def run(self, dispatcher, tracker, domain) -> List[Dict[Text, Any]]:
        country = tracker.get_slot('country')
        if not check_exist_country(country):
            dispatcher.utter_message("No Country with this name in our dataset")
            return []
        
        json_data = requests.post(self.__api_link , json.dumps({'country': country})).json()
        capital = json_data["body"]["population"]
        response = "population is {}".format(capital)
        dispatcher.utter_message(response)
        return []
