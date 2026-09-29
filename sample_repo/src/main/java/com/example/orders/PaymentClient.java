package com.example.orders;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
public class PaymentClient {

    private final RestClient restClient;

    public PaymentClient(RestClient.Builder builder) {
        // Connect and read timeouts are set on the payment RestClient in application.yml.
        this.restClient = builder.baseUrl("https://payments.internal").build();
    }

    @Retry(name = "paymentRetry")
    @CircuitBreaker(name = "paymentBreaker", fallbackMethod = "chargeFallback")
    public PaymentResult charge(String customerId, long amountCents) {
        return restClient.post()
                .uri("/v1/charges")
                .body(new ChargeRequest(customerId, amountCents))
                .retrieve()
                .body(PaymentResult.class);
    }

    // Called when the circuit is open or retries are exhausted. Fails fast instead of hanging.
    private PaymentResult chargeFallback(String customerId, long amountCents, Throwable cause) {
        throw new PaymentUnavailableException("Payment service unavailable, try again later", cause);
    }
}
