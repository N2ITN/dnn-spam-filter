from __future__ import print_function

import base64
import email
import json
import os

import httplib2
from apiclient import discovery, errors
from oauth2client import client, tools
from oauth2client.file import Storage


def pretty(json_mappable):
    print(json.dumps(json_mappable, sort_keys=True, indent=4, ensure_ascii=False), '\n' * 4)


try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/gmail-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
CLIENT_SECRET_FILE = 'client_id.json'
APPLICATION_NAME = 'Gmail API Python Quickstart'

import bs4


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, 'gmail-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else:  # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials


class query():
    """Shows basic usage of the Gmail API.

    Creates a Gmail API service object and outputs a list of label names
    of the user's Gmail account.
    """

    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)

    def get_labels(name='Recruiter'):
        results = query.service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        print([L['name'] for L in labels])

        for label in labels:
            if label['name'] == name:
                return label

    def get_messages():
        user_id = 'me'
        label_ids = query.get_labels(query.target)['id']
        idList = []

        def iter_res():
            response = query.service.users().messages().list(
                userId=user_id, maxResults=100, labelIds=label_ids,
                q='-{notifications@github.com}').execute()
            if 'messages' in response:

                idList.extend(response['messages'])

            if not 'nextPageToken' in response:

                yield response['messages']
            else:
                while 'nextPageToken' in response:
                    page_token = response['nextPageToken']
                    # print(page_token)
                    response = query.service.users().messages().list(
                        userId=user_id,
                        labelIds=label_ids,
                        q='-{notifications@github.com}',
                        pageToken=page_token,
                        maxResults=100).execute()

                    yield response['messages']

        idList = (next(iter_res()))
        # print(idList)
        for m in idList:
            r = m['id']

            # yield from query.metadata(r, user_id)
            yield from query.process(r, user_id)

    def metadata(msg_id, user_id):
        from dateutil import parser
        meta = query.service.users().messages().get(
            userId=user_id, format='metadata', id=msg_id).execute()
        headers = {h['name']: h['value'] for h in meta['payload']['headers']}
        pretty(list(headers.keys()))
        try:
            date = parser.parse(headers['Date'])
        except ValueError:
            fix = ' '.join(headers['Date'].rsplit()[:-1])
            date = parser.parse(fix)

        t = next((headers[x] for x in ['To', 'Delivered-To'] if x in headers))
        if "<" in t:
            t = t.split("<")[1].split(">")[0]
        f = headers['From']
        d = {
            'hour': date.hour,
            'weekday': date.weekday(),
            'tz': date.tzname(),
            'to': t,
            'from': f[f.find("<") + 1:f.find(">")],
            'subject': headers['Subject']
        }

        yield d

    def process(msg_id, user_id):

        full = query.service.users().messages().get(userId=user_id, id=msg_id).execute()
        mimeType = full['payload']['mimeType']
        print(mimeType)

        try:
            fullString = full['payload']['body']['data']
        except KeyError:

            parts = full['payload']['parts']
            print('parts', len(parts))
            if len(parts) < 2:
                n = 0
            else:
                n = 1
            mimeType = full['payload']['parts'][n]['mimeType']
            if 'text' in mimeType and 'mixed' not in mimeType:
                fullString = full['payload']['parts'][n]['body']['data']
            else:
                print('********************************************************************************')
                return
            # pretty(['payload']['parts'][1])

            # print(fullString)

        msg_str = base64.urlsafe_b64decode(fullString.encode('ascii'))

        b = bs4.BeautifulSoup(msg_str, 'html5lib').text
        b = b.split('}')[-1].replace('\n', ' ').replace('\t', ' ')

        c = [x for x in full['payload']['headers']]

        c = [
            n for n in c
            if n['name'] in ['To', 'Subject', 'Mime-Version', 'From', 'Date', 'Delivered-To']
        ]

        stop_words = [
            'the', 'that', 'to', 'as', 'at', 'there', 'has', 'and', 'or', 'is', 'not', 'a', 'of', 'but',
            'in', 'by', 'on', 'are', 'it', 'if'
        ]
        b = ''.join([n if n.isalpha() or n == ' ' else ' ' for n in b])

        e = ' '.join([w for w in b.split() if w not in stop_words and len(w) > 3])

        c = {kv['name']: kv['value'] for kv in c if kv['name'] in ['To', 'From', 'Subject', 'Date']}
        c['Subject'] = c['Subject'].split()

        if 'To' in c:
            c['To'] = c['To'].split('<')
        print(c['Date'])
        try:
            c["Time"] = c['Date'].split(':')[0].split(' ')[-1]
            print(c["Time"])
            c.pop('Date')
            c["From"] = c["From"].split('@')[1]
            yield e, c
        except Exception as ex:
            print('aw jeez')

            yield e, c


if __name__ == '__main__':
    import to_json

    def make_it_so(orders):
        targ, lim = orders
        query.limit = lim
        query.target = targ
        print(query.limit, query.target)
        to_json.set_json([b for b in query.get_messages()], query.target)

    targets = {'INBOX': 0}, {'RCRT': 100}
    for x in targets:
        orders = [[k, v] for k, v in x.items()]
        [make_it_so(i) for i in orders]
