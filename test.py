import configparser 

config = configparser.ConfigParser()
config.read('config.ini')

for s in config['DEFAULT']['_SESSION1'].split():
    print(s)