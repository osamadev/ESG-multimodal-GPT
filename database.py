import streamlit as st
from deta import Deta
import pandas as pd
import bcrypt

def hash_password(password):
    # Adding the salt to password
    salt = bcrypt.gensalt()
    # Hashing the password
    hashed_password = bcrypt.hashpw(str.encode(password), salt)
    return hashed_password.decode(), salt.decode()

def hash_password_with_salt(password: str, salt: str):
    # Hashing the password
    hashed_password = bcrypt.hashpw(str.encode(password), str.encode(salt))
    return hashed_password.decode()

# Load the environment variables
DETA_KEY = st.secrets["database_key"]

# Initialize with a project key
deta = Deta(DETA_KEY)

# This is how to create/connect a database
db = deta.Base("users_db")

def insert_user(email, name, password):
    """Returns the user on a successful user creation, otherwise raises an error"""
    hashed_password, salt = hash_password(password)
    return db.put({"email": email.lower(), "email": email, "name": name, 
            "password": hashed_password, "salt": salt}, key=email)

def fetch_all_users():
    """Returns a dict of all users"""
    res = db.fetch()
    return res.items

def is_user_exist(email):
    if get_user(email.lower()) is not None:
        return True
    return False

def login_user(email, password):
    user = db.get(email.lower())
    if user is not None:
        name = user["name"]
        hashed_password = user["password"]
        hash_salt = user["salt"]
        hashed_pw = hash_password_with_salt(password, hash_salt)
        if hashed_password == hashed_pw:
            return True, name, email
        else:
            return False, None, None
    return False, None, None

def get_user(email):
    """If not found, the function will return None"""
    return db.get(email)


def update_user(username, updates):
    """If the item is updated, returns None. Otherwise, an exception is raised"""
    return db.update(updates, username)


def delete_user(username):
    """Always returns None, even if the key does not exist"""
    return db.delete(username)
