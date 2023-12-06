FROM openjdk:8

WORKDIR /app

COPY target/your-spark-app.jar /app/your-spark-app.jar

CMD ["java", "-jar", "your-spark-app.jar"]
