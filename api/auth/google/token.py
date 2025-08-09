#!/usr/bin/env python3
"""
Google OAuth Token Exchange - Vercel Serverless Function
Handles OAuth code exchange for Google authentication
"""

import os
import json
import requests
from urllib.parse import urlencode

def handler(request):
    """Vercel serverless function handler"""
    
    # CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Content-Type': 'application/json'
    }
    
    # Handle preflight
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'headers': headers,
            'body': json.dumps({'error': 'Method not allowed'})
        }
    
    try:
        # Parse request body
        if hasattr(request, 'body'):
            body = json.loads(request.body.decode('utf-8') if isinstance(request.body, bytes) else request.body)
        else:
            body = json.loads(request.get_json())
        
        code = body.get('code')
        redirect_uri = body.get('redirect_uri')
        
        if not code or not redirect_uri:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Missing code or redirect_uri'})
            }
        
        # Google OAuth configuration
        client_id = os.environ.get('GOOGLE_CLIENT_ID')
        client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': 'OAuth not configured'})
            }
        
        # Exchange code for token
        token_data = {
            'code': code,
            'client_id': client_id,
            'client_secret': client_secret,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        token_response = requests.post(
            'https://oauth2.googleapis.com/token',
            data=token_data,
            headers={'Accept': 'application/json'}
        )
        
        if not token_response.ok:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Token exchange failed'})
            }
        
        tokens = token_response.json()
        access_token = tokens.get('access_token')
        
        if not access_token:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'No access token received'})
            }
        
        # Get user info
        user_response = requests.get(
            'https://www.googleapis.com/oauth2/v2/userinfo',
            headers={'Authorization': f'Bearer {access_token}'}
        )
        
        if not user_response.ok:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Failed to get user info'})
            }
        
        user_data = user_response.json()
        
        # Return user profile
        user_profile = {
            'id': user_data.get('id'),
            'name': user_data.get('name'),
            'email': user_data.get('email'),
            'avatar': user_data.get('picture'),
            'provider': 'google',
            'accessToken': access_token,
            'verified': user_data.get('verified_email', False)
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(user_profile)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }

# For direct invocation (testing)
if __name__ == '__main__':
    print("Google OAuth token exchange handler")