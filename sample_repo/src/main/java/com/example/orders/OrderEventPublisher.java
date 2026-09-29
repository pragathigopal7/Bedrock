package com.example.orders;

import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

@Component
public class OrderEventPublisher {

    private static final String TOPIC = "order-events";
    private static final String DEAD_LETTER_TOPIC = "order-events.dlq";

    private final KafkaTemplate<String, String> kafka;
    private final JsonCodec json;

    public OrderEventPublisher(KafkaTemplate<String, String> kafka, JsonCodec json) {
        this.kafka = kafka;
        this.json = json;
    }

    // The order id is the Kafka message key, so all events for one order land on the same
    // partition and are consumed in order.
    public void publish(String eventType, Order order) {
        String payload = json.write(new OrderEvent(eventType, order));
        kafka.send(TOPIC, order.id(), payload).whenComplete((result, error) -> {
            if (error != null) {
                // Events that fail to publish are sent to a dead letter topic for later replay.
                kafka.send(DEAD_LETTER_TOPIC, order.id(), payload);
            }
        });
    }
}
