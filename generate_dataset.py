"""
Dataset Generator v2: 5000 Train + 1000 Test
=============================================
- All 10 intents balanced (~500 actions each in train)
- ~33% multi-intent rows
- ~15% null rate per param for slot-filling training
- Test set uses DIFFERENT command templates to measure generalization
- No command overlap between train and test
"""

import json
import re
import random
import pandas as pd
from collections import Counter
from typing import Optional

random.seed(42)

# ============================================================
# PARAMETER POOLS (expanded from your existing 56 names, 40 locations, etc.)
# ============================================================
NAMES = [
    "Amit", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Rohan", "Kavitha",
    "Deepak", "Meera", "Arjun", "Neha", "Suresh", "Pooja", "Karthik", "Divya",
    "Manish", "Swati", "Rajesh", "Lakshmi", "Nikhil", "Isha", "Paarth", "Zara",
    "Farid", "Chen", "Yuki", "Omar", "Elena", "Marcus", "Sofia", "James",
    "Aarav", "Tina", "Uday", "Varun", "Zoe", "Diana", "Leo", "Sam",
    "Hina", "Jai", "Kavya", "Esha", "Chetan", "Nora", "Riya", "Kunal",
    "Vivek", "Megha", "Tanvi", "Ishaan", "Aditi", "Sahil", "Mira", "Dev",
    "Arun", "Bhavna", "Gaurav", "Jaya", "Kiran", "Leela", "Mohan", "Nandini",
    "Preeti", "Ramesh", "Sanjay", "Uma", "Wasim", "Xena", "Yogesh", "Rekha",
    "Ajay", "Geeta", "Harish", "Indira", "Jatin", "Kishore", "Madhavi", "Naveen",
    "Padma", "Raghu", "Shanti", "Tarun", "Urmila", "Venkat", "Aisha", "Bilal",
    "Carlos", "Dmitri", "Eklavya", "Fiona", "Gopal", "Hannah", "Ibrahim", "Julia",
    "Krishnamurthy", "Raghunath", "Jean-Pierre", "Dr. Sharma", "Prof. Anand",
]

LOCATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
    "Guwahati", "Pune", "Jaipur", "Ahmedabad", "New York", "London",
    "Tokyo", "Singapore", "Paris", "Sydney", "Berlin", "Dubai",
    "Bangkok", "Seoul", "Lucknow", "Austin", "Miami", "Seattle",
    "San Francisco", "Los Angeles", "Chicago", "Toronto", "Vancouver",
    "Chandigarh", "Bhopal", "Patna", "Kochi", "Mysore", "Nagpur",
    "Shimla", "Darjeeling", "Manali", "Gangtok", "Shillong",
]

TIMES = [
    "7 AM", "7:30 AM", "8 AM", "8:30 AM", "9 AM", "9:30 AM", "10 AM",
    "10:30 AM", "11 AM", "11:30 AM", "12 PM", "12:30 PM", "1 PM",
    "1:30 PM", "2 PM", "2:30 PM", "3 PM", "3:30 PM", "4 PM", "4:30 PM",
    "5 PM", "5:30 PM", "6 PM", "6:30 PM", "7 PM", "8 PM", "9 PM", "10 PM",
    "noon", "midnight", "morning", "afternoon", "evening",
    "14:00", "15:30", "17:00", "09:00", "10:00", "16:00", "18:00",
]

DATES = [
    "today", "tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday", "next week", "next Monday", "next Tuesday",
    "next Wednesday", "next Thursday", "next Friday", "this weekend",
    "January 10", "January 20", "February 14", "March 5", "March 15",
    "March 20", "April 1", "April 10", "May 1", "May 5", "June 10",
    "June 30", "July 4", "August 15", "September 1", "October 10",
    "November 25", "December 25", "December 31",
    "2026-04-01", "2026-05-15", "2026-06-30",
]

SUBJECTS = [
    "Project Update", "Meeting Notes", "Quarterly Review", "Budget Report",
    "Client Feedback", "Team Outing", "Weekly Sync", "Deadline Reminder",
    "New Proposal", "Sprint Planning", "Code Review Request", "Design Review",
    "Thank You Note", "Deployment Schedule", "Invoice", "Event Details",
    "Meeting Agenda", "Holiday Schedule", "Performance Review", "Training Schedule",
    "Status Update", "Action Items", "Follow Up", "Introduction",
    "Feedback Request", "Collaboration Request", "Approval Needed", "Agenda",
    "Team Standup Notes", "Resource Allocation",
]

BODIES = [
    "Here are the details we discussed in the meeting.",
    "Just a quick reminder about the upcoming deadline.",
    "I hope this email finds you well.",
    "Kindly review and approve the attached document.",
    "Please let me know your availability.",
    "Looking forward to your response.",
    "Please find the attached report for your review.",
    "Thanks for your help with this.",
    "Can we discuss this further in our next meeting?",
    "I wanted to follow up on our previous conversation.",
    "Sharing the updated document as discussed.",
    "Please confirm your attendance.",
    "Here is the summary of today's discussion.",
    "Let me know if you need any changes.",
    "Attaching the revised proposal for review.",
    "Please prioritize this task.",
    "Would appreciate your feedback on this.",
    "This is to inform you about the schedule change.",
    "Requesting your approval on the following.",
    "Here is the agenda for our upcoming meeting.",
]

SONGS = [
    "Bohemian Rhapsody", "Shape of You", "Blinding Lights", "Perfect",
    "Tum Hi Ho", "Chaiyya Chaiyya", "Kal Ho Naa Ho", "Senorita",
    "Levitating", "Starboy", "Closer", "Faded", "Despacito",
    "Someone Like You", "Rolling in the Deep", "Believer",
    "Viva La Vida", "Radioactive", "Thunder", "Dynamite",
    "Watermelon Sugar", "Bad Guy", "Sunflower", "Stay",
    "Tere Bina", "Kun Faya Kun", "Kesariya", "Raataan Lambiyan",
]

ARTISTS = [
    "Queen", "Ed Sheeran", "The Weeknd", "Arijit Singh", "AR Rahman",
    "Lata Mangeshkar", "Coldplay", "Taylor Swift", "Drake", "Adele",
    "Imagine Dragons", "Dua Lipa", "Alan Walker", "Kishore Kumar",
    "Shreya Ghoshal", "Sonu Nigam", "BTS", "Billie Eilish",
    "Post Malone", "Harry Styles", "Pritam", "Vishal Shekhar",
]

SEARCH_QUERIES = [
    "best restaurants near me", "Python tutorial for beginners",
    "latest news in AI", "how to fix a flat tire",
    "weather forecast for this weekend", "cheap flights to Goa",
    "machine learning interview questions", "healthy dinner recipes",
    "IIT Guwahati admission process", "stock market today",
    "best laptops under 50000", "history of the internet",
    "how to learn guitar", "top universities in India",
    "cryptocurrency trends", "home workout routines",
    "best books on deep learning", "visa requirements for Japan",
    "upcoming cricket matches", "how to write a research paper",
    "electric cars comparison 2026", "yoga for beginners",
    "best cafes in Guwahati", "how to prepare for GATE exam",
    "latest smartphone reviews", "organic farming techniques",
]

TASKS = [
    "Submit the project report", "Buy groceries", "Call the dentist",
    "Review pull request", "Update resume", "Pay electricity bill",
    "Prepare presentation slides", "Book flight tickets",
    "Clean the apartment", "Reply to client emails",
    "Finish homework", "Order birthday cake", "Renew gym membership",
    "Schedule car service", "Return library books",
    "Water the plants", "Backup laptop data", "Pick up dry cleaning",
    "Send the invoice", "Complete the assignment",
    "Book hotel for trip", "Refill prescriptions", "Fix the bug in code",
    "Submit tax documents", "Organize desk",
]

PRIORITIES = ["high", "medium", "low"]

ALARM_LABELS = [
    "Morning Run", "Wake Up", "Take Medicine", "Team Standup",
    "Lunch Break", "Pick Up Kids", "Gym Time", "Study Session",
    "Meditation", "Evening Walk", "Dinner Prep", "Bedtime",
    "Water Plants", "Daily Review", "Stretch Break", "Reading Time",
]

PICKUP_LOCATIONS = [
    "Airport", "Railway Station", "Home", "Office", "Mall",
    "Bus Stand", "Hotel Marriott", "IIT Campus", "City Center",
    "Metro Station", "Hospital", "University Gate", "Paltan Bazaar",
    "GS Road", "Zoo Road", "Maligaon", "Dispur", "Chandmari",
]

DESTINATIONS = [
    "Hotel Grand", "Airport", "Office", "Home", "Railway Station",
    "Convention Center", "Hospital", "University", "Restaurant",
    "Shopping Mall", "Client Office", "Tech Park", "Bus Stand",
    "City Center", "IIT Campus", "Lokpriya Gopinath Bordoloi Airport",
    "Guwahati Club", "Kamakhya Temple", "Saraighat Bridge",
]

MESSAGES = [
    "I will be late today", "Meeting has been rescheduled",
    "Please send the report", "Happy birthday!",
    "Can you call me back?", "On my way",
    "Let's catch up this weekend", "Don't forget the deadline",
    "Thanks for your help", "See you at the meeting",
    "Running 10 minutes late", "Lunch at 1?",
    "The presentation is ready", "Check your email",
    "I have sent the documents", "Can we reschedule?",
    "Great work on the project", "Please confirm the booking",
    "I will handle it", "Let me know when you're free",
]


# ============================================================
# TEMPLATE POOLS — TRAIN vs TEST use DIFFERENT phrasings
# ============================================================

TEMPLATES = {
    "send_email": {
        "train": [
            "Send an email to {name} about {subject} saying {body}",
            "Email {name} about {subject}",
            "Mail {name} regarding {subject} with message {body}",
            "Compose an email to {name} regarding {subject} with the message {body}",
            "Shoot an email over to {name} about {subject}",
            "Can you send {name} an email about {subject}",
            "Please email {name} with the subject line {subject} and write {body}",
            "Send a message to {name} via email about {subject} saying {body}",
            "Draft an email for {name} about {subject}",
            "Write an email to {name} about {subject} and say {body}",
            "Fire off an email to {name} regarding {subject}",
            "Forward an email to {name} with subject {subject}",
            "Send an email about {subject}",
            "Email {name} regarding {subject} saying {body}",
            "Drop {name} an email about {subject}",
        ],
        "test": [
            "Ping {name} via email about {subject}",
            "Shoot {name} a mail on {subject} with message {body}",
            "Could you email {name} regarding {subject}",
            "I need to email {name} about {subject} and tell them {body}",
            "Dispatch an email to {name} with subject {subject}",
            "Quickly email {name} about {subject}",
            "Send {name} a note via email about {subject} saying {body}",
            "Notify {name} by email about {subject}",
        ],
    },
    "schedule_meeting": {
        "train": [
            "Schedule a meeting with {name} on {date} at {time}",
            "Book a meeting with {name} at {time} on {date}",
            "Set up a meeting with {name} on {date} at {time}",
            "Arrange a call with {name} at {time} on {date}",
            "Put a meeting on my calendar with {name} at {time} on {date}",
            "Organize a session with {name} at {time} on {date}",
            "Can you schedule a meeting with {name} on {date} at {time}",
            "Fix a meeting with {name} on {date}",
            "Plan a meeting on {date}",
            "Book a meeting on {date}",
            "Create a meeting invite for {name} on {date} at {time}",
            "I want to meet {name} on {date} at {time}",
            "Set a meeting with {name} at {time} on {date}",
            "Block time for a meeting with {name} on {date}",
            "Reserve a slot for meeting {name} on {date} at {time}",
        ],
        "test": [
            "Lock in a meeting with {name} for {date} at {time}",
            "Pencil in {name} on {date} at {time}",
            "Could you book a session with {name} on {date}",
            "Add a meeting with {name} on {date} at {time} to my calendar",
            "Get me a meeting slot with {name} on {date}",
            "I'd like to meet {name} at {time} on {date}",
            "Confirm a meeting with {name} for {date} at {time}",
            "Put {name} on my schedule for {date} at {time}",
        ],
    },
    "get_weather": {
        "train": [
            "What's the weather in {location} on {date}",
            "Check the weather in {location}",
            "How's the weather in {location} on {date}",
            "What is the weather like in {location}",
            "Weather forecast for {location} on {date}",
            "How is the weather in {location} on {date}",
            "What does the weather look like in {location} on {date}",
            "Look up the weather for {location} on {date}",
            "Tell me the weather in {location}",
            "Check the weather",
            "How's the weather on {date}",
            "Get me the weather report for {location}",
            "What's the forecast for {location} on {date}",
            "Is it going to rain in {location} on {date}",
        ],
        "test": [
            "Any idea about the weather in {location}",
            "Pull up the forecast for {location} on {date}",
            "Will it be sunny in {location} on {date}",
            "Show me {location} weather for {date}",
            "I need the weather update for {location}",
            "What's the climate like in {location} on {date}",
            "Give me the weather in {location} for {date}",
        ],
    },
    "set_reminder": {
        "train": [
            "Set a reminder to {task} at {time} on {date}",
            "Remind me to {task} at {time}",
            "Create a reminder for {task} on {date} at {time}",
            "Add a reminder to {task} on {date}",
            "Don't let me forget to {task} at {time}",
            "I need a reminder to {task} at {time} on {date}",
            "Can you remind me to {task} on {date}",
            "Please set a reminder to {task} at {time}",
            "Set up a reminder for {task} on {date}",
            "Remind me on {date} at {time} to {task}",
            "Put a reminder for {task} at {time}",
            "Make a reminder to {task} on {date}",
        ],
        "test": [
            "Alert me to {task} at {time} on {date}",
            "I want a reminder about {task} on {date}",
            "Nudge me to {task} at {time}",
            "Schedule a reminder for {task} on {date} at {time}",
            "Remember to tell me to {task} at {time}",
            "Pop a reminder for {task} at {time} on {date}",
        ],
    },
    "send_message": {
        "train": [
            "Send a message to {name} saying {message}",
            "Text {name} that {message}",
            "Message {name}: {message}",
            "Drop a message to {name} saying {message}",
            "Send {name} a text saying {message}",
            "Let {name} know that {message}",
            "Ping {name} with the message {message}",
            "Tell {name} that {message}",
            "Text {name}: {message}",
            "Shoot a message to {name} saying {message}",
            "DM {name} saying {message}",
            "Notify {name} that {message}",
        ],
        "test": [
            "Drop {name} a line saying {message}",
            "Buzz {name} with {message}",
            "Forward a text to {name}: {message}",
            "Pass a message to {name} that {message}",
            "Reach out to {name} saying {message}",
            "Hit up {name} with {message}",
        ],
    },
    "set_alarm": {
        "train": [
            "Set an alarm for {time} labeled {label}",
            "Set alarm at {time} called {label}",
            "Wake me up at {time}",
            "Create an alarm for {time} with label {label}",
            "I need an alarm at {time} named {label}",
            "Alarm at {time} for {label}",
            "Set a {time} alarm labeled {label}",
            "Add an alarm at {time} tagged {label}",
            "Ring the alarm at {time} for {label}",
            "Set my alarm to {time}",
            "Put an alarm at {time} labeled {label}",
            "Schedule an alarm for {time} called {label}",
        ],
        "test": [
            "Buzz me at {time} for {label}",
            "Set a wake-up call for {time}",
            "Alarm me at {time} tagged {label}",
            "I want an alarm ringing at {time} for {label}",
            "Configure an alarm at {time} labeled {label}",
            "Please wake me at {time}",
        ],
    },
    "book_cab": {
        "train": [
            "Book a cab from {pickup} to {destination} at {time}",
            "Get me a ride from {pickup} to {destination}",
            "I need a taxi from {pickup} to {destination} at {time}",
            "Call a cab to {destination} from {pickup}",
            "Arrange a cab from {pickup} to {destination} at {time}",
            "Book an Uber from {pickup} to {destination}",
            "Order a ride from {pickup} to {destination} at {time}",
            "Get a cab from {pickup} to {destination}",
            "Hail a taxi from {pickup} to {destination} at {time}",
            "Book a ride to {destination} from {pickup}",
            "Reserve a cab from {pickup} to {destination} at {time}",
            "I want a cab from {pickup} to {destination}",
        ],
        "test": [
            "Grab a cab from {pickup} to {destination} at {time}",
            "Summon a ride from {pickup} headed to {destination}",
            "Fetch me a taxi from {pickup} going to {destination}",
            "Arrange transport from {pickup} to {destination} at {time}",
            "Need a ride from {pickup} dropping at {destination}",
            "Request a cab from {pickup} to {destination}",
        ],
    },
    "play_music": {
        "train": [
            "Play {song} by {artist}",
            "Put on {song} by {artist}",
            "I want to listen to {song} by {artist}",
            "Play the song {song}",
            "Can you play {song} by {artist}",
            "Start playing {song} by {artist}",
            "Queue up {song} by {artist}",
            "Play me some {artist}",
            "Turn on {song} by {artist}",
            "I'd like to hear {song} by {artist}",
            "Blast {song} by {artist}",
            "Shuffle {artist} songs",
        ],
        "test": [
            "Fire up {song} by {artist}",
            "Cue {song} from {artist}",
            "Get {song} by {artist} going",
            "Spin {song} by {artist}",
            "Throw on some {artist}",
            "Stream {song} by {artist}",
        ],
    },
    "search_web": {
        "train": [
            "Search for {query}",
            "Google {query}",
            "Look up {query}",
            "Find information about {query}",
            "Search the web for {query}",
            "Can you search {query}",
            "I want to know about {query}",
            "What can you find about {query}",
            "Search online for {query}",
            "Find me results for {query}",
            "Do a web search for {query}",
            "Look for {query} online",
        ],
        "test": [
            "Hunt down info on {query}",
            "Research {query} for me",
            "Dig up {query} on the web",
            "Query the internet for {query}",
            "Help me find {query}",
            "Scour the web for {query}",
        ],
    },
    "create_task": {
        "train": [
            "Create a task to {title} due {due_date} priority {priority}",
            "Add a to-do: {title} due {due_date}",
            "Make a task for {title} due {due_date} priority {priority}",
            "Add {title} to my task list due {due_date}",
            "Create a to-do item: {title} priority {priority}",
            "I need to {title} due {due_date}",
            "Add task {title} due {due_date} priority {priority}",
            "New task: {title} due {due_date}",
            "Put {title} on my to-do list priority {priority}",
            "Schedule a task to {title} by {due_date}",
            "Assign myself the task {title} due {due_date}",
            "Log a task: {title} due {due_date} priority {priority}",
        ],
        "test": [
            "Jot down a task to {title} due {due_date}",
            "Throw {title} on my task board priority {priority}",
            "Note a to-do: {title} by {due_date}",
            "Set up a task for {title} due {due_date} priority {priority}",
            "Track task {title} due {due_date}",
            "Pencil in a task to {title} by {due_date}",
        ],
    },
}

# Multi-intent templates
MULTI_TEMPLATES = {
    "train": [
        ("send_email", "schedule_meeting",
         "Email {name1} about {subject} and schedule a meeting with {name2} on {date} at {time}"),
        ("send_email", "schedule_meeting",
         "Mail {name1} regarding {subject} and book a call with {name2} at {time} on {date}"),
        ("send_email", "schedule_meeting",
         "Send an email to {name1} about {subject} and set up a meeting with {name2} on {date}"),
        ("send_email", "get_weather",
         "Email {name1} about {subject} and check the weather in {location} on {date}"),
        ("send_email", "get_weather",
         "Send a mail to {name1} regarding {subject} and look up weather for {location}"),
        ("schedule_meeting", "get_weather",
         "Schedule a meeting with {name1} on {date} at {time} and check weather in {location}"),
        ("schedule_meeting", "get_weather",
         "Book a meeting with {name1} at {time} on {date} and how's the weather in {location}"),
        ("send_email", "set_reminder",
         "Email {name1} about {subject} and set a reminder to {task} at {time}"),
        ("send_email", "set_reminder",
         "Mail {name1} regarding {subject} and remind me to {task} on {date}"),
        ("schedule_meeting", "set_reminder",
         "Schedule a meeting with {name1} on {date} and remind me to {task} at {time}"),
        ("set_alarm", "get_weather",
         "Set an alarm for {time} labeled {label} and check weather in {location}"),
        ("book_cab", "schedule_meeting",
         "Book a cab from {pickup} to {destination} and schedule a meeting with {name1} at {time}"),
        ("book_cab", "send_message",
         "Book a cab from {pickup} to {destination} and text {name1} that {message}"),
        ("play_music", "set_alarm",
         "Set an alarm for {time} and play {song} by {artist}"),
        ("play_music", "get_weather",
         "Play {song} by {artist} and check the weather in {location}"),
        ("send_message", "set_reminder",
         "Text {name1} that {message} and remind me to {task} at {time}"),
        ("search_web", "send_email",
         "Search for {query} and email {name1} about {subject}"),
        ("create_task", "send_email",
         "Create a task to {title} due {due_date} and email {name1} about {subject}"),
        ("create_task", "set_reminder",
         "Add a task to {title} due {due_date} and remind me to {task} at {time}"),
        ("send_message", "schedule_meeting",
         "Message {name1} that {message} and book a meeting with {name2} on {date}"),
    ],
    "test": [
        ("send_email", "schedule_meeting",
         "Drop a mail to {name1} on {subject} and lock in a meeting with {name2} for {date} at {time}"),
        ("send_email", "get_weather",
         "Ping {name1} about {subject} via email and pull up the forecast for {location}"),
        ("schedule_meeting", "get_weather",
         "Pencil in {name1} on {date} at {time} and any idea about weather in {location}"),
        ("set_reminder", "send_email",
         "Alert me to {task} at {time} and notify {name1} by email about {subject}"),
        ("book_cab", "schedule_meeting",
         "Grab a cab from {pickup} to {destination} and confirm a meeting with {name1} at {time}"),
        ("play_music", "get_weather",
         "Fire up {song} by {artist} and show me weather in {location}"),
        ("send_message", "set_reminder",
         "Buzz {name1} with {message} and nudge me to {task} at {time}"),
        ("create_task", "send_email",
         "Jot down a task to {title} and shoot {name1} an email about {subject}"),
        ("search_web", "send_message",
         "Research {query} and drop {name1} a line saying {message}"),
        ("set_alarm", "book_cab",
         "Buzz me at {time} for {label} and arrange transport from {pickup} to {destination}"),
    ],
}


# ============================================================
# GENERATION HELPERS
# ============================================================
def rand_or_null(pool, null_prob=0.15):
    return None if random.random() < null_prob else random.choice(pool)


def make_action(intent, params):
    return {"intent": intent, "parameters": params}


def build_params(intent, null_prob=0.15):
    """Build parameter dict for an intent with random values and null rate."""
    if intent == "send_email":
        return {"to": rand_or_null(NAMES, null_prob), "subject": rand_or_null(SUBJECTS, null_prob), "body": rand_or_null(BODIES, null_prob)}
    elif intent == "schedule_meeting":
        return {"person": rand_or_null(NAMES, null_prob), "time": rand_or_null(TIMES, null_prob), "date": rand_or_null(DATES, null_prob)}
    elif intent == "get_weather":
        return {"location": rand_or_null(LOCATIONS, null_prob), "date": rand_or_null(DATES, null_prob)}
    elif intent == "set_reminder":
        return {"task": rand_or_null(TASKS, null_prob), "time": rand_or_null(TIMES, null_prob), "date": rand_or_null(DATES, null_prob)}
    elif intent == "send_message":
        return {"to": rand_or_null(NAMES, null_prob), "message": rand_or_null(MESSAGES, null_prob)}
    elif intent == "set_alarm":
        return {"time": rand_or_null(TIMES, null_prob), "label": rand_or_null(ALARM_LABELS, null_prob)}
    elif intent == "book_cab":
        return {"pickup": rand_or_null(PICKUP_LOCATIONS, null_prob), "destination": rand_or_null(DESTINATIONS, null_prob), "time": rand_or_null(TIMES, null_prob)}
    elif intent == "play_music":
        return {"song": rand_or_null(SONGS, null_prob), "artist": rand_or_null(ARTISTS, null_prob)}
    elif intent == "search_web":
        return {"query": rand_or_null(SEARCH_QUERIES, null_prob)}
    elif intent == "create_task":
        return {"title": rand_or_null(TASKS, null_prob), "due_date": rand_or_null(DATES, null_prob), "priority": rand_or_null(PRIORITIES, null_prob)}
    return {}


def fill_template(template, params, intent, extra_params=None):
    """Fill a template string with parameter values, handling nulls gracefully."""
    all_params = dict(params)
    if extra_params:
        all_params.update(extra_params)

    # Build substitution dict with fallback removal of null phrases
    subs = {}
    for k, v in all_params.items():
        subs[k] = v if v is not None else ""

    # Add convenience keys
    if "name" not in subs:
        subs["name"] = subs.get("to") or subs.get("person") or ""
    if "title" not in subs and "task" in subs:
        subs["title"] = subs["task"]

    try:
        result = template.format(**subs)
    except KeyError:
        return None

    # Clean up artifacts from null substitutions
    result = re.sub(r"\s+(?:about|regarding|saying|at|on|due|priority|labeled?|called|named|tagged|from|to)\s*$", "", result)
    result = re.sub(r"\s+(?:about|regarding|saying|at|on|due|priority|labeled?|called|named|tagged)\s+(?:about|regarding|saying|at|on|due|priority|labeled?|called|named|tagged)", " ", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    result = result.rstrip(".,;:!? ")

    if len(result) < 10:
        return None
    return result


# ============================================================
# SINGLE-INTENT ROW GENERATOR
# ============================================================
def generate_single_intent_rows(intent, count, split="train"):
    rows = []
    templates = TEMPLATES[intent][split]

    for _ in range(count * 3):  # oversample, then truncate
        if len(rows) >= count:
            break

        params = build_params(intent, null_prob=0.15)
        template = random.choice(templates)

        # Build substitution-friendly params
        subs = {}
        for k, v in params.items():
            subs[k] = v if v is not None else ""

        # Add convenience aliases
        subs.setdefault("name", subs.get("to", "") or subs.get("person", ""))
        subs.setdefault("title", subs.get("task", ""))

        try:
            cmd = template.format(**subs)
        except KeyError:
            continue

        # Clean nulled-out phrases
        cmd = re.sub(r"\s+(?:about|regarding|saying|at|on|due|priority|labeled?|called|named|tagged|from|to|with)\s*$", "", cmd)
        cmd = re.sub(r"\s{2,}", " ", cmd).strip().rstrip(".,;:!? ")

        if len(cmd) < 10:
            continue

        action = make_action(intent, params)
        api_call = json.dumps({"actions": [action]})
        rows.append({"command": cmd, "api_call": api_call})

    return rows[:count]


# ============================================================
# MULTI-INTENT ROW GENERATOR
# ============================================================
def generate_multi_intent_rows(count, split="train"):
    rows = []
    templates = MULTI_TEMPLATES[split]

    for _ in range(count * 3):
        if len(rows) >= count:
            break

        intent1, intent2, template = random.choice(templates)
        params1 = build_params(intent1, null_prob=0.12)
        params2 = build_params(intent2, null_prob=0.12)

        # Build unified substitution dict
        subs = {}
        for k, v in params1.items():
            subs[k] = v if v is not None else ""
        for k, v in params2.items():
            if k not in subs:
                subs[k] = v if v is not None else ""

        # Name handling for multi-intent
        name_pool = random.sample(NAMES, min(4, len(NAMES)))
        subs.setdefault("name1", subs.get("to", "") or subs.get("person", "") or name_pool[0])
        subs.setdefault("name2", name_pool[1] if len(name_pool) > 1 else name_pool[0])
        subs.setdefault("name", subs["name1"])
        subs.setdefault("title", subs.get("task", ""))

        # Ensure name params in actions match template names
        if "to" in params1:
            params1["to"] = subs["name1"] if subs["name1"] else None
        if "person" in params1:
            params1["person"] = subs["name1"] if subs["name1"] else None
        if "to" in params2:
            params2["to"] = subs.get("name2", subs["name1"]) if subs.get("name2") or subs.get("name1") else None
        if "person" in params2:
            params2["person"] = subs.get("name2", subs["name1"]) if subs.get("name2") or subs.get("name1") else None

        try:
            cmd = template.format(**subs)
        except KeyError:
            continue

        cmd = re.sub(r"\s+(?:about|regarding|saying|at|on|due|priority|labeled?|called|named|tagged|from|to|with)\s*$", "", cmd)
        cmd = re.sub(r"\s{2,}", " ", cmd).strip().rstrip(".,;:!? ")

        if len(cmd) < 15:
            continue

        actions = [make_action(intent1, params1), make_action(intent2, params2)]
        api_call = json.dumps({"actions": actions})
        rows.append({"command": cmd, "api_call": api_call})

    return rows[:count]


# ============================================================
# MAIN GENERATOR
# ============================================================
def generate_dataset():
    all_intents = list(TEMPLATES.keys())

    # --- TRAIN SET: 5000 rows ---
    # Target: ~335 single-intent per intent (3350 total) + 1650 multi-intent
    train_rows = []

    print("Generating TRAIN set (5000 rows)...")
    for intent in all_intents:
        rows = generate_single_intent_rows(intent, 335, split="train")
        train_rows.extend(rows)
        print(f"  {intent}: {len(rows)} single-intent rows")

    multi_train = generate_multi_intent_rows(1650, split="train")
    train_rows.extend(multi_train)
    print(f"  multi-intent: {len(multi_train)} rows")

    # Pad to exactly 5000 if needed
    while len(train_rows) < 5000:
        intent = random.choice(all_intents)
        extra = generate_single_intent_rows(intent, 10, split="train")
        train_rows.extend(extra)
    train_rows = train_rows[:5000]

    # --- TEST SET: 1000 rows (different templates) ---
    test_rows = []

    print("\nGenerating TEST set (1000 rows)...")
    for intent in all_intents:
        rows = generate_single_intent_rows(intent, 80, split="test")
        test_rows.extend(rows)
        print(f"  {intent}: {len(rows)} single-intent rows")

    multi_test = generate_multi_intent_rows(200, split="test")
    test_rows.extend(multi_test)
    print(f"  multi-intent: {len(multi_test)} rows")

    while len(test_rows) < 1000:
        intent = random.choice(all_intents)
        extra = generate_single_intent_rows(intent, 10, split="test")
        test_rows.extend(extra)
    test_rows = test_rows[:1000]

    # --- Build DataFrames ---
    train_df = pd.DataFrame(train_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sample(frac=1, random_state=43).reset_index(drop=True)

    # --- Verify ---
    for name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        intent_counts = Counter()
        multi = 0
        null_total = 0
        param_total = 0
        for i in range(len(df)):
            obj = json.loads(df.iloc[i]["api_call"])
            if len(obj["actions"]) > 1:
                multi += 1
            for a in obj["actions"]:
                intent_counts[a["intent"]] += 1
                for k, v in a["parameters"].items():
                    param_total += 1
                    if v is None:
                        null_total += 1

        print(f"\n{name} SET ({len(df)} rows):")
        print(f"  Multi-intent: {multi} ({multi/len(df)*100:.1f}%)")
        print(f"  Null rate: {null_total}/{param_total} ({null_total/max(param_total,1)*100:.1f}%)")
        print(f"  Intent distribution:")
        for k, v in intent_counts.most_common():
            print(f"    {k}: {v}")

    # --- Check overlap ---
    train_cmds = set(train_df["command"].str.lower())
    test_cmds = set(test_df["command"].str.lower())
    overlap = train_cmds & test_cmds
    print(f"\nCommand overlap: {len(overlap)} (should be 0 or near 0)")

    # --- Save ---
    train_df.to_csv("train_dataset_5k.csv", index=False)
    test_df.to_csv("test_dataset_1k.csv", index=False)
    print(f"\nSaved: train_dataset_5k.csv ({len(train_df)} rows)")
    print(f"Saved: test_dataset_1k.csv ({len(test_df)} rows)")


if __name__ == "__main__":
    generate_dataset()
