package com.example.orders;

import java.time.Duration;
import java.util.Optional;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

@Component
public class IdempotencyStore {

    // Keys expire after 24 hours, which bounds Redis memory while covering realistic client retries.
    private static final Duration KEY_TTL = Duration.ofHours(24);

    private final StringRedisTemplate redis;
    private final JsonCodec json;

    public IdempotencyStore(StringRedisTemplate redis, JsonCodec json) {
        this.redis = redis;
        this.json = json;
    }

    public Optional<OrderResponse> find(String key) {
        String stored = redis.opsForValue().get("idem:" + key);
        return Optional.ofNullable(stored).map(s -> json.read(s, OrderResponse.class));
    }

    public void save(String key, OrderResponse response) {
        redis.opsForValue().set("idem:" + key, json.write(response), KEY_TTL);
    }
}
