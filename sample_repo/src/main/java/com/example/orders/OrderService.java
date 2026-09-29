package com.example.orders;

import java.util.Optional;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class OrderService {

    private final IdempotencyStore idempotencyStore;
    private final PaymentClient paymentClient;
    private final OrderRepository orderRepository;
    private final OrderEventPublisher eventPublisher;

    public OrderService(IdempotencyStore idempotencyStore,
                        PaymentClient paymentClient,
                        OrderRepository orderRepository,
                        OrderEventPublisher eventPublisher) {
        this.idempotencyStore = idempotencyStore;
        this.paymentClient = paymentClient;
        this.orderRepository = orderRepository;
        this.eventPublisher = eventPublisher;
    }

    @Transactional
    public OrderResponse createOrder(OrderRequest request, String idempotencyKey) {
        // Step 1: if this key was already processed, return the stored response instead of reprocessing.
        Optional<OrderResponse> previous = idempotencyStore.find(idempotencyKey);
        if (previous.isPresent()) {
            return previous.get();
        }
        // Step 2: validate the request. Orders must have at least one line item and a positive total.
        if (request.items().isEmpty() || request.totalCents() <= 0) {
            throw new InvalidOrderException("Order needs at least one item and a positive total");
        }
        // Step 3: charge the customer through the payment service.
        PaymentResult payment = paymentClient.charge(request.customerId(), request.totalCents());
        // Step 4: persist the order, then publish an ORDER_CREATED event to Kafka.
        Order order = orderRepository.save(Order.from(request, payment.transactionId()));
        eventPublisher.publish("ORDER_CREATED", order);
        OrderResponse response = OrderResponse.from(order);
        idempotencyStore.save(idempotencyKey, response);
        return response;
    }

    public Optional<OrderResponse> findOrder(String id) {
        return orderRepository.findById(id).map(OrderResponse::from);
    }

    @Transactional
    public void cancelOrder(String id) {
        Order order = orderRepository.findById(id)
                .orElseThrow(() -> new OrderNotFoundException(id));
        // Shipped orders cannot be cancelled; only PENDING and PAID orders can.
        if (order.status() == OrderStatus.SHIPPED) {
            throw new InvalidOrderException("Shipped orders cannot be cancelled");
        }
        orderRepository.save(order.withStatus(OrderStatus.CANCELLED));
        eventPublisher.publish("ORDER_CANCELLED", order);
    }
}
