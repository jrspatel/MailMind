import os 
import pickle 
from datetime import datetime
import json
import google.auth.transport.requests 
from google_auth_oauthlib.flow import InstalledAppFlow 
from googleapiclient.discovery import build 

# define the scope for gmail api access 
scopes = ['https://www.googleapis.com/auth/gmail.readonly'] 


def authenticate_gmail():
    creds = None
    # re-usable user credemtials, doesn't need to authenticate everytime
    if os.path.exists(''):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token) 
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', scopes)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds



def build_gmail():
    credentials = authenticate_gmail() 
    service = build('gmail', 'v1', credentials=credentials) 
    return service 

def list_messages(service, user_id='me', query=''):
    """
    List Gmail messages based on a query.
    :param service: Authenticated Gmail API service instance.
    :param user_id: User ID (default is 'me' for authenticated user).
    :param query: Search query (e.g., 'is:unread').
    """
    print("service:", service)
    results = service.users().messages().list(userId=user_id, q=query).execute()
    messages = results.get('messages', [])
    print(f"Found {len(messages)} messages.")
    return messages

def get_message_details(service, msg_id, user_id='me'):
    """
    Get details of a Gmail message.
    :param service: Authenticated Gmail API service instance.
    :param msg_id: Message ID.
    :param user_id: User ID (default is 'me').
    """
    message = service.users().messages().get(userId=user_id, id=msg_id).execute()
    payload = message['payload']
    headers = payload.get('headers', [])
    snippet = message.get('snippet', '')

    # Extract subject and sender information
    subject = sender = receiver = ''
    for header in headers:
        if header['name'] == 'Subject':
            subject = header['value']
        if header['name'] == 'From':
            sender = header['value']
        if header['name'] == 'Delivered-To':
            receiver = header['value']
    
    timestamp_ms = int(message.get('internalDate', 0))
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        "id": msg_id,
        "subject": subject,
        "sender": sender,
        "receiver": receiver,
        "timestamp": timestamp,
        "snippet": snippet
    }




if __name__ == '__main__':
    service = build_gmail()
    messages = list_messages(service, query='is:unread')  # Fetch unread messages

    email_data = []

    if messages:
        for msg in messages:  # Display details for the first 5 messages
            email = get_message_details(service, msg['id'])
            email_data.append(email) 
    
    with open('emails.json','w') as json_file:
        json.dump(email_data, json_file, indent=4)