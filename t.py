# Create function that sends GET request to https://nv.api.intsurfing.com/name-validator/validate with a name parameter and x-api-key header, and returns the validation result.
# call the function 100 times with different names

from time import sleep
import requests


def validate_name(name):
    url = "https://nv.api.intsurfing.com/name-validator/validate"
    headers = {"x-api-key": "B6D2T5gwwo51LyUSw8moz5e76poOQovk3jKt0hTW"}
    params = {"name": name}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}


firstnames = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
    "Karl",
    "Leo",
    "Mallory",
    "Nina",
    "Oscar",
    "Peggy",
    "Quentin",
    "Ruth",
    "Sam",
    "Trudy",
    "Uma",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zach",
]

lastnames = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Sanchez",
    "Clark",
    "Ramirez",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
]

requests_count = 0
for i in range(500):
    for firstname in firstnames:
        for lastname in lastnames:
            fullname = f"{firstname} {lastname}"
            result = validate_name(fullname)
            # results.append(result)

            requests_count += 1
            print(requests_count, ":::::", result)
            # sleep(1)  # Add a delay to avoid hitting rate limits
