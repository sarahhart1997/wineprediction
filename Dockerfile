FROM openjdk:11-jre-slim

WORKDIR /app

COPY target/wineproj-1.0-SNAPSHOT.jar /app/

CMD ["java", "-cp", "/app/wineproj-1.0-SNAPSHOT.jar", "njit.sarah.wineproject.WineQualityTraining"]
