#Base Image to use
FROM python:3.7.9

#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt Contextual_Search_App/requirements.txt

#install all requirements in requirements.txt
RUN pip3 install -r Contextual_Search_App/requirements.txt

#Copy all files in current directory into app directory
COPY . /Contextual_Search_App

#Change Working Directory to app directory
WORKDIR /Contextual_Search_App

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "Contextual_Search_App.py", "--server.port=8080", "--server.address=0.0.0.0"]
