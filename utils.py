import os 
import yaml

def get_apikey():
    """
    Reads API key from a configuration file.

    This function opens a configuration file named "apikeys.yml", reads the 
    API key for OpenAI

    Returns: api_key (str): The OpenAI API key.
    """
    
    # Construct the full path to the configuration file
    script_dir = "/usercode/"
    file_path = os.path.join(script_dir, "apikeys.yml")

    with open(file_path, 'r') as yamlfile:
        loaded_yamlfile = yaml.safe_load(yamlfile)
        API_KEY = loaded_yamlfile['openai']['api_key']
        
    return API_KEY

if __name__ == "__main__":
    print("API_KEY", get_apikey())