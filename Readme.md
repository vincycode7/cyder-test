Solution:

1.  Q: Remove any personally identifiable information from the dataset

    A: To remove personally identifiable information, we need to identify the columns that contain such information. From the given data, the columns "userId" and "ip" contain personally identifiable information. We can drop these columns using the pandas library.

2.  Q: Using the data, derive any demographic or interest-based attributes about the individual

    A: To derive demographic or interest-based attributes, we need to analyze the remaining columns of the dataset. We can use techniques like data visualization, statistical analysis, and machine learning algorithms to derive such attributes.

3.  Q: Identify products the person is interested in.

    A: We can use natural language processing techniques to analyze the metadata content column to identify products the person is interested in.

4.  Q: Explain how you would know if this person is interested in a new product.

    A: We can track the frequency and recency of the person's visits to a particular product's website to determine if the person is interested in a new product.

5.  Q: Is this person interested in or was browsing financial products? (Loans, Credit Cards, Mortgages, etc.)

    A: We can use the domains visited by the person to identify if they were interested in or browsing financial products.

6.  Q: How would you improve accuracy if you had more time

    A: We can improve accuracy by collecting more data, using more advanced machine learning algorithms, and performing feature engineering to extract more relevant features from the data.

7.  Q: What other data would you want collected that chrome extensions can collect that can enforce the model.

    A: 
        
        i. Browser history: This would allow the model to gain insights into the user's browsing behavior, which could inform the types of ads they are likely to engage with.

        ii. Time spent on pages: By tracking how long users spend on particular pages, the model could infer how engaged they are with different types of content and adjust its recommendations accordingly.

        iii. Clickstream data: By analyzing the links that users click on, the model could gain insights into their interests and use that information to make more personalized ad recommendations.

        iv. Search history: By analyzing the user's search history, the model could gain insights into their intent and use that information to make more targeted ad recommendations.

        v. Social media activity: By monitoring the user's social media activity, the model could gain insights into their interests, relationships, and social network, which could inform more personalized ad recommendations.

1. install airflow
2. open a terminial
3. create a .env file using the .env.exmple format, the set the following variables
    i. AIRFLOW_HOME=/absolute/path/to/the/project/locally
    ii. AIRFLOW__CORE__DAGS_FOLDERS=${AIRFLOW_HOME}/dags
    iii. AIRFLOW__CORE__PLUGINS_FOLDERS=${AIRFLOW_HOME}/plugins`

2. if this your first time running this project type `init db` in terminal, to initiallize the database
3. if this is your first time running this  project type  `airflow users create --username admin --firstname FIRST_NAME --lastname LAST_NAME --role Admin --email admin@example.org` to create admin user
4. run with `airflow webserver -p 8080` 
5. open a second terminal while the first is still running
6. run `airflow scheduler`
7. login into UI with your username and password which are `admin` and `admin`
