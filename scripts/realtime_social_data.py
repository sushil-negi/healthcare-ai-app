#!/usr/bin/env python3
"""
Real-time Social Media Data Pipeline
Demonstrates streaming data collection for continuous model training
"""

import json
import os
import random
import time
from datetime import datetime


class RealtimeDataPipeline:
    """
    Simulates real-time data streaming from social media platforms
    In production, this would connect to actual APIs
    """

    def __init__(self):
        self.platforms = ["facebook", "twitter", "linkedin", "instagram"]
        self.data_buffer = []
        self.buffer_size = 100

    def generate_realtime_post(self):
        """Generate a realistic social media post"""

        topics = [
            "AI",
            "machine learning",
            "cloud computing",
            "cybersecurity",
            "blockchain",
            "IoT",
            "5G",
            "quantum computing",
            "robotics",
            "data science",
            "DevOps",
            "automation",
            "digital transformation",
        ]

        templates = [
            "Just launched our new {topic} solution! Check it out: [link]",
            "Excited about the future of {topic}. What are your thoughts?",
            "{topic} is transforming how we work. Here's what we learned...",
            "Breaking: Major advancement in {topic} announced today!",
            "How {topic} can help your business grow in 2024",
            "5 ways {topic} is changing the industry",
            "Our team just implemented {topic} - amazing results!",
            "Why every company needs to understand {topic}",
            "The truth about {topic} that nobody talks about",
            "Success story: How we used {topic} to increase efficiency by 50%",
        ]

        hashtags = [
            "#tech",
            "#innovation",
            "#AI",
            "#futureofwork",
            "#digital",
            "#transformation",
            "#startup",
            "#technology",
            "#business",
        ]

        # Generate post
        topic = random.choice(topics)
        template = random.choice(templates)
        text = template.format(topic=topic)

        # Add hashtags
        num_hashtags = random.randint(1, 3)
        selected_hashtags = random.sample(hashtags, num_hashtags)
        text += " " + " ".join(selected_hashtags)

        return {
            "id": f"post_{int(time.time())}_{random.randint(1000, 9999)}",
            "platform": random.choice(self.platforms),
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "engagement": {
                "likes": random.randint(10, 1000),
                "shares": random.randint(0, 100),
                "comments": random.randint(0, 50),
            },
        }

    def stream_data(self, duration_seconds=60):
        """Stream data for specified duration"""

        print(f"🔄 Starting real-time data stream for {duration_seconds} seconds...")
        print("=" * 60)

        start_time = time.time()
        posts_collected = 0

        while time.time() - start_time < duration_seconds:
            # Generate 1-5 posts per second (simulating real-time flow)
            posts_per_batch = random.randint(1, 5)

            for _ in range(posts_per_batch):
                post = self.generate_realtime_post()
                self.data_buffer.append(post)
                posts_collected += 1

                # Print sample posts
                if posts_collected % 10 == 0:
                    print(f"📝 [{post['platform']}] {post['text'][:60]}...")
                    print(
                        f"   👍 {post['engagement']['likes']} likes, "
                        f"🔄 {post['engagement']['shares']} shares"
                    )

            # Process buffer if full
            if len(self.data_buffer) >= self.buffer_size:
                self.process_buffer()

            # Simulate real-time delay
            time.sleep(1)

        # Process remaining data
        if self.data_buffer:
            self.process_buffer()

        print(f"\n✅ Collected {posts_collected} posts in {duration_seconds} seconds")
        return posts_collected

    def process_buffer(self):
        """Process buffered data for training"""

        if not self.data_buffer:
            return

        # Extract text for training
        training_texts = []

        for post in self.data_buffer:
            # Original text
            training_texts.append(post["text"])

            # High engagement posts get extra weight
            if post["engagement"]["likes"] > 500:
                training_texts.append(f"Popular: {post['text']}")

            # Platform-specific formatting
            if post["platform"] == "twitter":
                training_texts.append(f"Tweet: {post['text'][:280]}")
            elif post["platform"] == "linkedin":
                training_texts.append(f"Professional update: {post['text']}")

        # Save batch to file
        batch_file = f"data/realtime_batch_{int(time.time())}.json"
        os.makedirs("data", exist_ok=True)

        with open(batch_file, "w") as f:
            json.dump(
                {
                    "texts": training_texts,
                    "metadata": {
                        "batch_size": len(self.data_buffer),
                        "timestamp": datetime.now().isoformat(),
                        "platforms": list(set(p["platform"] for p in self.data_buffer)),
                    },
                },
                f,
            )

        print(f"\n💾 Saved batch of {len(self.data_buffer)} posts to {batch_file}")

        # Clear buffer
        self.data_buffer = []

    def continuous_training_demo(self):
        """Demonstrate continuous model training with streaming data"""

        print(
            """
🔄 Continuous Learning Pipeline Demo
===================================

This demonstrates how to:
1. Stream real-time social media data
2. Process data in batches
3. Continuously update the model
4. Monitor performance in MLflow

In production, this would:
- Connect to real APIs (Facebook, Twitter, etc.)
- Use message queues (Kafka, RabbitMQ)
- Implement data validation
- Handle rate limits and errors
- Store in data lake for replay
        """
        )

        # Simulate streaming for 30 seconds
        posts = self.stream_data(30)

        print(f"\n📊 Streaming Summary:")
        print(f"   Total posts: {posts}")
        print(f"   Average rate: {posts/30:.1f} posts/second")
        print(
            f"   Data files created: {len(os.listdir('data')) if os.path.exists('data') else 0}"
        )


def setup_realtime_pipeline():
    """Guide for production real-time setup"""

    print(
        """
🚀 Production Real-time Pipeline Setup
=====================================

1. **Data Sources**
   ```python
   # Facebook Graph API Webhooks
   webhook_url = "https://your-api.com/facebook/webhook"
   
   # Twitter Streaming API
   twitter_stream = TwitterStream(bearer_token='your-token')
   
   # LinkedIn API
   linkedin_api = LinkedIn(client_id='your-id')
   ```

2. **Message Queue Setup**
   ```yaml
   # Kafka for high-throughput
   kafka:
     brokers: ["localhost:9092"]
     topics:
       - social-media-posts
       - user-interactions
       - model-feedback
   ```

3. **Stream Processing**
   ```python
   # Apache Flink or Spark Streaming
   stream = (
       kafka_source
       .filter(validate_post)
       .map(preprocess_text)
       .window(minutes=5)
       .reduce(aggregate_batches)
   )
   ```

4. **Continuous Training**
   ```python
   # Online learning with mini-batches
   for batch in stream:
       model.partial_fit(batch)
       mlflow.log_metrics(model.score(batch))
   ```

5. **Data Privacy Compliance**
   - Anonymize user data
   - Implement data retention policies
   - GDPR/CCPA compliance
   - Secure data transmission

6. **Rate Limit Management**
   - Facebook: 200 calls/hour
   - Twitter: 300 requests/15-min
   - LinkedIn: 100 calls/day
   - Implement backoff strategies

7. **Error Handling**
   - Retry failed requests
   - Dead letter queues
   - Monitoring and alerting
   - Graceful degradation
    """
    )


if __name__ == "__main__":
    # Show setup guide
    setup_realtime_pipeline()

    # Run demo
    print("\n" + "=" * 60)
    print("🎬 Running Real-time Data Pipeline Demo")
    print("=" * 60)

    pipeline = RealtimeDataPipeline()
    pipeline.continuous_training_demo()

    print("\n✅ Demo complete!")
    print("\n📚 To use with your model:")
    print("1. Review generated data in 'data/' directory")
    print("2. Combine batches for training")
    print("3. Implement continuous learning loop")
    print("4. Monitor model drift in MLflow")
