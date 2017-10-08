# dnn-spam-filter
Gmail API and Keras neural net for spam classification



`gmail_api` connects to gmail and the function `def get_messages` filters by certain tags.

you will need a `client_id.json` that is set up like this:
```

{"installed":{"client_id":"","project_id":"sodium-dynamo-164521","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://accounts.google.com/o/oauth2/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"","redirect_uris":["urn:ietf:wg:oauth:2.0:oob","http://localhost"]}}

```