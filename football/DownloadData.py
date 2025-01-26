import requests
from bs4 import BeautifulSoup
import os

websites = ["englandm", "scotlandm", "germanym", "italym", "spainm", "francem", "netherlandsm", "belgiumm", "portugalm", "turkeym", "greecem"]
count_file = 1
for country in websites:
    url = f"https://www.football-data.co.uk/{country}.php"
    response = requests.get(url)

    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        csv_links = soup.find_all('a', href=True)
        csv_urls = ['https://www.football-data.co.uk/' + link['href'] for link in csv_links if link['href'].endswith('.csv')]
        
        if not os.path.exists('csv_files'):
            os.makedirs('csv_files')
        
        count = 1  # Initialize counter
        for csv_url in csv_urls:
            if count > 30:
                break  # Exit loop if count exceeds 80
            csv_response = requests.get(csv_url)
            if csv_response.status_code == 200:

                csv_filename = os.path.join('csv_files', f"{count_file}.csv")
                with open(csv_filename, 'wb') as file:
                    file.write(csv_response.content)
                print(f'Downloaded: {csv_filename}')
                count += 1
                count_file +=1
            else:
                print(f'Failed to download {csv_url}')
    else:
        print(f'Failed to retrieve the webpage. Status code: {response.status_code}')


