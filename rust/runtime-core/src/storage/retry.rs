// rust/runtime-core/src/storage/retry.rs

//! Retry policy configuration for storage backends.
//!
//! This module provides a configurable retry policy with exponential backoff
//! and jitter for handling transient failures in storage operations.

use std::time::Duration;

use crate::config::S3Config;

/// Retry policy configuration.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay between retries.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Backoff multiplier (e.g., 2.0 for exponential backoff).
    pub backoff_multiplier: f64,
    /// Whether to add random jitter to delays.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl From<&S3Config> for RetryConfig {
    fn from(s3_config: &S3Config) -> Self {
        Self {
            max_retries: s3_config.max_retries,
            initial_delay: Duration::from_millis(s3_config.retry_delay_ms),
            max_delay: Duration::from_millis(s3_config.max_retry_delay_ms),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Creates a new retry configuration with custom settings.
    pub fn new(
        max_retries: u32,
        initial_delay: Duration,
        max_delay: Duration,
        backoff_multiplier: f64,
    ) -> Self {
        Self {
            max_retries,
            initial_delay,
            max_delay,
            backoff_multiplier,
            jitter: true,
        }
    }

    /// Creates a retry configuration with no retries.
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Creates a retry configuration for aggressive retry (many retries, short delays).
    pub fn aggressive() -> Self {
        Self {
            max_retries: 10,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 1.5,
            jitter: true,
        }
    }

    /// Creates a retry configuration for conservative retry (few retries, longer delays).
    pub fn conservative() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 3.0,
            jitter: true,
        }
    }

    /// Calculates the delay for a given attempt number (0-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt >= self.max_retries {
            return Duration::ZERO;
        }

        let base_delay = self.initial_delay.as_secs_f64()
            * self.backoff_multiplier.powi(attempt as i32);

        let delay_secs = base_delay.min(self.max_delay.as_secs_f64());

        let final_delay = if self.jitter {
            // Add up to 25% jitter
            let jitter_factor = 1.0 + (rand_simple(attempt) * 0.25);
            delay_secs * jitter_factor
        } else {
            delay_secs
        };

        Duration::from_secs_f64(final_delay)
    }

    /// Returns true if more retries are allowed for the given attempt.
    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_retries
    }
}

/// Simple deterministic pseudo-random number generator for jitter.
/// Uses the attempt number as seed to produce a value in [0, 1).
fn rand_simple(seed: u32) -> f64 {
    // Simple LCG-based PRNG
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (x as f64) / (u32::MAX as f64)
}

/// Represents a retryable operation result.
#[derive(Debug)]
pub enum RetryResult<T, E> {
    /// Operation succeeded.
    Ok(T),
    /// Operation failed but can be retried.
    Retry(E),
    /// Operation failed and should not be retried.
    Fail(E),
}

impl<T, E> RetryResult<T, E> {
    /// Returns true if the result is Ok.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }

    /// Returns true if the operation should be retried.
    pub fn should_retry(&self) -> bool {
        matches!(self, Self::Retry(_))
    }

    /// Converts to a standard Result, discarding retry information.
    pub fn into_result(self) -> Result<T, E> {
        match self {
            Self::Ok(v) => Ok(v),
            Self::Retry(e) | Self::Fail(e) => Err(e),
        }
    }
}

/// Execute an async operation with retries.
pub async fn retry_async<T, E, F, Fut>(
    config: &RetryConfig,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = RetryResult<T, E>>,
{
    let mut attempt = 0;

    loop {
        match operation().await {
            RetryResult::Ok(value) => return Ok(value),
            RetryResult::Fail(error) => return Err(error),
            RetryResult::Retry(error) => {
                if !config.should_retry(attempt) {
                    return Err(error);
                }

                let delay = config.delay_for_attempt(attempt);
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay, Duration::from_millis(100));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(config.jitter);
    }

    #[test]
    fn test_no_retry() {
        let config = RetryConfig::no_retry();
        assert_eq!(config.max_retries, 0);
        assert!(!config.should_retry(0));
    }

    #[test]
    fn test_should_retry() {
        let config = RetryConfig::default();
        assert!(config.should_retry(0));
        assert!(config.should_retry(4));
        assert!(!config.should_retry(5));
        assert!(!config.should_retry(100));
    }

    #[test]
    fn test_delay_for_attempt() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: false, // Disable jitter for predictable testing
        };

        // Without jitter, delays should be exact
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800));
    }

    #[test]
    fn test_delay_respects_max() {
        let config = RetryConfig {
            max_retries: 10,
            initial_delay: Duration::from_secs(10),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: false,
        };

        // Should be capped at max_delay
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(30));
        assert_eq!(config.delay_for_attempt(5), Duration::from_secs(30));
    }

    #[test]
    fn test_delay_with_jitter() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        };

        let delay = config.delay_for_attempt(0);
        // With jitter, delay should be between 100ms and 125ms
        assert!(delay >= Duration::from_millis(100));
        assert!(delay <= Duration::from_millis(125));
    }

    #[test]
    fn test_from_s3_config() {
        let s3_config = S3Config {
            max_retries: 3,
            retry_delay_ms: 200,
            max_retry_delay_ms: 5000,
            ..Default::default()
        };

        let retry_config = RetryConfig::from(&s3_config);
        assert_eq!(retry_config.max_retries, 3);
        assert_eq!(retry_config.initial_delay, Duration::from_millis(200));
        assert_eq!(retry_config.max_delay, Duration::from_millis(5000));
    }

    #[test]
    fn test_retry_result() {
        let ok: RetryResult<i32, &str> = RetryResult::Ok(42);
        assert!(ok.is_ok());
        assert!(!ok.should_retry());
        assert_eq!(ok.into_result(), Ok(42));

        let retry: RetryResult<i32, &str> = RetryResult::Retry("error");
        assert!(!retry.is_ok());
        assert!(retry.should_retry());
        assert_eq!(retry.into_result(), Err("error"));

        let fail: RetryResult<i32, &str> = RetryResult::Fail("fatal");
        assert!(!fail.is_ok());
        assert!(!fail.should_retry());
        assert_eq!(fail.into_result(), Err("fatal"));
    }

    #[tokio::test]
    async fn test_retry_async_success() {
        let config = RetryConfig::default();
        let result = retry_async(&config, || async {
            RetryResult::Ok::<_, &str>(42)
        }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_retry_async_fail() {
        let config = RetryConfig::default();
        let result = retry_async(&config, || async {
            RetryResult::Fail::<i32, _>("fatal error")
        }).await;
        assert_eq!(result, Err("fatal error"));
    }

    #[tokio::test]
    async fn test_retry_async_eventual_success() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 1.0,
            jitter: false,
        };

        let attempts = std::sync::atomic::AtomicU32::new(0);
        let result = retry_async(&config, || {
            let count = attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if count < 2 {
                    RetryResult::Retry::<i32, _>("not yet")
                } else {
                    RetryResult::Ok(42)
                }
            }
        }).await;

        assert_eq!(result, Ok(42));
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_async_exhausted() {
        let config = RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 1.0,
            jitter: false,
        };

        let attempts = std::sync::atomic::AtomicU32::new(0);
        let result = retry_async(&config, || {
            attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async { RetryResult::Retry::<i32, _>("always fails") }
        }).await;

        assert_eq!(result, Err("always fails"));
        // Initial attempt + 2 retries = 3 total attempts
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 3);
    }
}
