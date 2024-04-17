import requests
import logging


class APIHandler:
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url
        self.api_token = api_token

    def make_api_call(self, content: str, **params) -> requests.Response:
        """
        Makes a POST request to the API endpoint and returns the response.

        Parameters
        ----------
        content : str
            The content type for the API call.
        **params : any
            Additional parameters to pass in the API call according to the content type.
            Refer to the API documentation for more information: https://redcap.raras.org.br/api/help/

        Returns
        -------
        requests.Response
            The HTTP response object.
        """
        # Setup payload
        payload = {
            'token': self.api_token,
            'content': content
        }
        payload.update(params)

        # Make request
        response = requests.post(self.api_url, data=payload, timeout=30, verify=True)
        logging.info('HTTP Status: %s', response.status_code)

        return response
