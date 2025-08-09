#!/usr/bin/env python3
"""
Session Refresh Handler - Vercel Serverless Function
Handles session refresh for authenticated users
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta

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
        
        user = body.get('user')
        
        if not user:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'No user data provided'})
            }
        
        provider = user.get('provider')
        access_token = user.get('accessToken') or user.get('apiKey')
        
        if not provider or not access_token:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Invalid user data'})
            }
        
        # Refresh user data based on provider
        if provider == 'google':
            refreshed_user = refresh_google_user(access_token)
        elif provider == 'github':
            refreshed_user = refresh_github_user(access_token)
        elif provider == 'openai':
            refreshed_user = refresh_openai_user(access_token)
        else:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': f'Unsupported provider: {provider}'})
            }
        
        if not refreshed_user:
            return {
                'statusCode': 401,
                'headers': headers,
                'body': json.dumps({'error': 'Failed to refresh user session'})
            }
        
        # Return refreshed user data
        response_data = {
            'user': refreshed_user,
            'expiresIn': 86400000,  # 24 hours
            'refreshedAt': int(time.time() * 1000)
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }

def refresh_google_user(access_token):
    """Refresh Google user data"""
    try:
        response = requests.get(
            'https://www.googleapis.com/oauth2/v2/userinfo',
            headers={'Authorization': f'Bearer {access_token}'}
        )
        
        if response.ok:
            user_data = response.json()
            return {
                'id': user_data.get('id'),
                'name': user_data.get('name'),
                'email': user_data.get('email'),
                'avatar': user_data.get('picture'),
                'provider': 'google',
                'accessToken': access_token,
                'verified': user_data.get('verified_email', False),
                'lastRefresh': int(time.time() * 1000)
            }
    except:
        pass
    return None

def refresh_github_user(access_token):
    """Refresh GitHub user data"""
    try:
        response = requests.get(
            'https://api.github.com/user',
            headers={'Authorization': f'token {access_token}'}
        )
        
        if response.ok:
            user_data = response.json()
            return {
                'id': user_data.get('id'),
                'name': user_data.get('name') or user_data.get('login'),
                'email': user_data.get('email'),
                'avatar': user_data.get('avatar_url'),
                'provider': 'github',
                'accessToken': access_token,
                'username': user_data.get('login'),
                'verified': True,
                'lastRefresh': int(time.time() * 1000)
            }
    except:
        pass
    return None

def refresh_openai_user(api_key):
    """Validate OpenAI API key"""
    try:
        response = requests.get(
            'https://api.openai.com/v1/models',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=10
        )
        
        if response.ok:
            return {
                'id': 'openai-user',
                'name': 'OpenAI User',
                'email': '',
                'provider': 'openai',
                'apiKey': api_key,
                'verified': True,
                'lastRefresh': int(time.time() * 1000)
            }
    except:
        pass
    return None

# For direct invocation (testing)
if __name__ == '__main__':
    print("Session refresh handler")