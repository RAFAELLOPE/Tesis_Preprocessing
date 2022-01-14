# Tesis_Preprocessing

### Databricks configuration
1. Install Databricks and Databricks-connect
```
pip install databricks databricks-connect
```

2. Configure Databricks
```
databricks configure
```
Databricks Host: the URL of your Databricks Workspace
Username: your username in that Workspace
Password: An [Access Token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) generated for your Workspace User

3. We can view the content of our databricks cli configuration file with this command:
```
less ~/.databrickscfg
```
4. List and specify cluster to use with Databricks Connect
```
databricks clusters list | grep Rafa
```

5. Test spark job

This may fail if you don't have JDK 8 installed. You can install the
open JDK 8 by visiting this link: https://adoptopenjdk.net/
```
park-submit *etl.py --username FILL_IN_YOUR_USERNAME
```

6. Command 'dbsf' of Databricks CLI:
```
dbfs cp <origin_file> dbfs:/<destination_file>
```