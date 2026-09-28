package id.poc.inappai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class InAppAiApplication {

    public static void main(String[] args) {
        SpringApplication.run(InAppAiApplication.class, args);
    }
}
