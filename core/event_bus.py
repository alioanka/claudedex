"""
Event Bus - Event-driven architecture for decoupled communication
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json
import uuid

class EventType(Enum):
    """System event types"""
    # Market events
    NEW_PAIR_DETECTED = "new_pair_detected"
    PRICE_UPDATE = "price_update"
    VOLUME_SURGE = "volume_surge"
    LIQUIDITY_CHANGE = "liquidity_change"
    
    # Trading events
    OPPORTUNITY_FOUND = "opportunity_found"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    
    # Risk events
    HIGH_RISK_DETECTED = "high_risk_detected"
    RUG_PULL_WARNING = "rug_pull_warning"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_HIT = "take_profit_hit"
    
    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    ERROR_OCCURRED = "error_occurred"
    WARNING_RAISED = "warning_raised"
    
    # ML events
    MODEL_RETRAINED = "model_retrained"
    PREDICTION_MADE = "prediction_made"
    PATTERN_DETECTED = "pattern_detected"
    
    # Portfolio events
    REBALANCE_NEEDED = "rebalance_needed"
    PORTFOLIO_UPDATED = "portfolio_updated"
    DAILY_LIMIT_REACHED = "daily_limit_reached"
    
    # Alert events
    ALERT_TRIGGERED = "alert_triggered"
    NOTIFICATION_SENT = "notification_sent"

@dataclass
class Event:
    """Event data structure"""
    event_type: EventType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[str] = None
    priority: int = 5  # 1-10, 1 being highest priority
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'priority': self.priority,
            'metadata': self.metadata
        }
        
    def to_json(self) -> str:
        """Convert event to JSON"""
        return json.dumps(self.to_dict(), default=str)

@dataclass
class EventSubscription:
    """Subscription to events"""
    subscriber_id: str
    event_type: EventType
    callback: Callable
    filters: Optional[Dict] = None
    priority: int = 5
    async_handler: bool = True
    
class EventBus:
    """Central event bus for system communication"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize event bus
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Subscribers registry
        self.subscribers: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        
        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
        # Event history
        self.event_history: List[Event] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # Statistics
        self.stats = {
            'events_emitted': 0,
            'events_processed': 0,
            'events_failed': 0,
            'by_type': defaultdict(int)
        }
        
        # Processing task
        self.processing_task = None
        self.is_running = False
        
        # Event filters
        self.global_filters = []
        
        # Dead letter queue for failed events
        self.dead_letter_queue: List[Tuple[Event, Exception]] = []
        
    async def start(self):
        """Start event bus processing"""
        if not self.is_running:
            self.is_running = True
            self.processing_task = asyncio.create_task(self._process_events())
            
            # Emit system started event
            await self.emit(Event(
                event_type=EventType.SYSTEM_STARTED,
                data={'timestamp': datetime.now()},
                source='EventBus'
            ))
            
    async def stop(self):
        """Stop event bus processing"""
        if self.is_running:
            self.is_running = False
            
            # Emit system stopping event
            await self.emit(Event(
                event_type=EventType.SYSTEM_STOPPED,
                data={'timestamp': datetime.now()},
                source='EventBus'
            ))
            
            # Wait for queue to empty
            await self.event_queue.join()
            
            # Cancel processing task
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
                    
    async def emit(self, event: Event):
        """
        Emit an event to all subscribers
        
        Args:
            event: Event to emit
        """
        try:
            # Apply global filters
            if not self._apply_filters(event, self.global_filters):
                return
                
            # Add to queue for async processing
            await self.event_queue.put(event)
            
            # Update statistics
            self.stats['events_emitted'] += 1
            self.stats['by_type'][event.event_type] += 1
            
            # Add to history
            self._add_to_history(event)
            
        except Exception as e:
            print(f"Error emitting event: {e}")
            
    # Rename the original sync methods to have different names to avoid confusion
    def subscribe_sync(self, event_type: EventType, callback: Callable,
                      subscriber_id: Optional[str] = None, filters: Optional[Dict] = None,
                      priority: int = 5) -> str:
        """
        Synchronous subscribe with full parameters (internal use)
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
            subscriber_id: Optional subscriber identifier
            filters: Optional filters to apply
            priority: Subscription priority (1-10)
            
        Returns:
            Subscription ID
        """
        try:
            if subscriber_id is None:
                subscriber_id = str(uuid.uuid4())
                
            subscription = EventSubscription(
                subscriber_id=subscriber_id,
                event_type=event_type,
                callback=callback,
                filters=filters,
                priority=priority,
                async_handler=asyncio.iscoroutinefunction(callback)
            )
            
            # Add to subscribers list (sorted by priority)
            self.subscribers[event_type].append(subscription)
            self.subscribers[event_type].sort(key=lambda x: x.priority)
            
            return subscriber_id
            
        except Exception as e:
            print(f"Subscription error: {e}")
            return ""
    
    def unsubscribe_sync(self, subscriber_id: str, event_type: Optional[EventType] = None):
        """
        Synchronous unsubscribe (internal use)
        
        Args:
            subscriber_id: Subscriber to remove
            event_type: Optional specific event type to unsubscribe from
        """
        try:
            if event_type:
                # Remove from specific event type
                self.subscribers[event_type] = [
                    sub for sub in self.subscribers[event_type]
                    if sub.subscriber_id != subscriber_id
                ]
            else:
                # Remove from all event types
                for event_type in self.subscribers:
                    self.subscribers[event_type] = [
                        sub for sub in self.subscribers[event_type]
                        if sub.subscriber_id != subscriber_id
                    ]
                    
        except Exception as e:
            print(f"Unsubscribe error: {e}")
            
    async def _process_events(self):
        """Process events from the queue"""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                # Process event
                await self._handle_event(event)
                
                # Mark as done
                self.event_queue.task_done()
                
                # Update statistics
                self.stats['events_processed'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Event processing error: {e}")
                self.stats['events_failed'] += 1
                
    async def _handle_event(self, event: Event):
        """
        Handle a single event
        
        Args:
            event: Event to handle
        """
        try:
            # Get subscribers for this event type
            subscribers = self.subscribers.get(event.event_type, [])
            
            # Process subscribers by priority
            for subscription in subscribers:
                try:
                    # Apply subscription filters
                    if subscription.filters and not self._apply_filters(event, subscription.filters):
                        continue
                        
                    # Call handler
                    if subscription.async_handler:
                        await subscription.callback(event)
                    else:
                        subscription.callback(event)
                        
                except Exception as e:
                    print(f"Subscriber {subscription.subscriber_id} error: {e}")
                    
                    # Add to dead letter queue
                    self.dead_letter_queue.append((event, e))
                    
                    # Emit error event
                    if event.event_type != EventType.ERROR_OCCURRED:  # Prevent infinite loop
                        await self.emit(Event(
                            event_type=EventType.ERROR_OCCURRED,
                            data={
                                'original_event': event.to_dict(),
                                'error': str(e),
                                'subscriber': subscription.subscriber_id
                            },
                            source='EventBus'
                        ))
                        
        except Exception as e:
            print(f"Event handling error: {e}")
            
    def _apply_filters(self, event: Event, filters: Any) -> bool:
        """
        Apply filters to an event
        
        Args:
            event: Event to filter
            filters: Filter criteria
            
        Returns:
            True if event passes filters
        """
        try:
            if not filters:
                return True
                
            if isinstance(filters, dict):
                for key, value in filters.items():
                    if hasattr(event, key):
                        if getattr(event, key) != value:
                            return False
                    elif key in event.data:
                        if event.data[key] != value:
                            return False
                    else:
                        return False
                        
            elif callable(filters):
                return filters(event)
                
            return True
            
        except Exception as e:
            print(f"Filter application error: {e}")
            return True
            
    def _add_to_history(self, event: Event):
        """Add event to history with size limit"""
        try:
            self.event_history.append(event)
            
            # Trim history if needed
            if len(self.event_history) > self.max_history_size:
                self.event_history = self.event_history[-self.max_history_size:]
                
        except Exception as e:
            print(f"History update error: {e}")
            
    def get_history(self, event_type: Optional[EventType] = None,
                   limit: int = 100) -> List[Event]:
        """
        Get event history
        
        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of historical events
        """
        try:
            if event_type:
                filtered = [e for e in self.event_history if e.event_type == event_type]
            else:
                filtered = self.event_history
                
            return filtered[-limit:]
            
        except Exception as e:
            print(f"History retrieval error: {e}")
            return []
            
    def get_statistics(self) -> Dict:
        """Get event bus statistics"""
        return {
            'events_emitted': self.stats['events_emitted'],
            'events_processed': self.stats['events_processed'],
            'events_failed': self.stats['events_failed'],
            'events_by_type': dict(self.stats['by_type']),
            'queue_size': self.event_queue.qsize(),
            'subscribers_count': sum(len(subs) for subs in self.subscribers.values()),
            'history_size': len(self.event_history),
            'dead_letter_queue_size': len(self.dead_letter_queue)
        }
        
# Update the wait_for method to use the sync version internally
    async def wait_for(self, event_type: EventType, timeout: Optional[float] = None,
                       filters: Optional[Dict] = None) -> Optional[Event]:
        """
        Wait for a specific event
        
        Args:
            event_type: Type of event to wait for
            timeout: Optional timeout in seconds
            filters: Optional filters to apply
            
        Returns:
            Event if received within timeout
        """
        try:
            future = asyncio.Future()
            
            def callback(event: Event):
                if not future.done():
                    future.set_result(event)
                    
            # Subscribe temporarily using sync method
            sub_id = self.subscribe_sync(event_type, callback, filters=filters)
            
            try:
                # Wait for event with timeout
                if timeout:
                    event = await asyncio.wait_for(future, timeout)
                else:
                    event = await future
                    
                return event
                
            except asyncio.TimeoutError:
                return None
                
            finally:
                # Unsubscribe using sync method
                self.unsubscribe_sync(sub_id, event_type)
                
        except Exception as e:
            print(f"Wait for event error: {e}")
            return None
            
    def clear_dead_letter_queue(self) -> List[Tuple[Event, Exception]]:
        """
        Clear and return dead letter queue
        
        Returns:
            List of failed events and their exceptions
        """
        dlq = self.dead_letter_queue.copy()
        self.dead_letter_queue.clear()
        return dlq
        
    def add_global_filter(self, filter_func: Callable[[Event], bool]):
        """
        Add a global filter for all events
        
        Args:
            filter_func: Function that returns True if event should be processed
        """
        self.global_filters.append(filter_func)
        
    def remove_global_filter(self, filter_func: Callable[[Event], bool]):
        """
        Remove a global filter
        
        Args:
            filter_func: Filter function to remove
        """
        if filter_func in self.global_filters:
            self.global_filters.remove(filter_func)
            
    async def replay_events(self, events: List[Event]):
        """
        Replay a list of events
        
        Args:
            events: Events to replay
        """
        try:
            for event in events:
                # Update timestamp to current
                event.timestamp = datetime.now()
                event.metadata['replayed'] = True
                
                # Emit event
                await self.emit(event)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"Event replay error: {e}")
            
    def create_event_stream(self, event_types: List[EventType]) -> asyncio.Queue:
        """
        Create a stream for specific event types
        
        Args:
            event_types: Types of events to stream
            
        Returns:
            Queue that receives matching events
        """
        try:
            stream_queue = asyncio.Queue()
            
            async def stream_handler(event: Event):
                await stream_queue.put(event)
                
            # Subscribe to all requested event types using sync method
            for event_type in event_types:
                self.subscribe_sync(event_type, stream_handler)
                
            return stream_queue
            
        except Exception as e:
            print(f"Stream creation error: {e}")
            return asyncio.Queue()

    # Add these wrapper methods to existing EventBus class
    
# Replace the duplicate async wrapper methods in EventBus class with these corrected versions
# Remove the existing async wrappers (lines ~440-490) and replace with:

    async def publish(self, event_type: str, data: Dict) -> None:
        """
        Publish event (async wrapper matching API spec)
        
        Args:
            event_type: Type of event as string
            data: Event data dictionary
        """
        try:
            # Convert string to EventType enum
            if hasattr(EventType, event_type.upper()):
                event_type_enum = getattr(EventType, event_type.upper())
            else:
                # Try direct value match
                for et in EventType:
                    if et.value == event_type:
                        event_type_enum = et
                        break
                else:
                    # Create a generic event type if not found
                    print(f"Unknown event type: {event_type}, using ERROR_OCCURRED")
                    event_type_enum = EventType.ERROR_OCCURRED
                    data['original_event_type'] = event_type
            
            # Create and emit event
            event = Event(
                event_type=event_type_enum,
                data=data,
                source='EventBus.publish'
            )
            await self.emit(event)
            
        except Exception as e:
            print(f"Publish error: {e}")
    
    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to events (async wrapper matching API spec)
        
        Args:
            event_type: Type of event as string
            handler: Callback function to handle events
        """
        try:
            # Convert string to EventType enum
            if hasattr(EventType, event_type.upper()):
                event_type_enum = getattr(EventType, event_type.upper())
            else:
                # Try direct value match
                for et in EventType:
                    if et.value == event_type:
                        event_type_enum = et
                        break
                else:
                    print(f"Unknown event type: {event_type}")
                    return
            
            # Generate unique subscriber ID for this handler
            subscriber_id = str(uuid.uuid4())
            
            # Create subscription
            subscription = EventSubscription(
                subscriber_id=subscriber_id,
                event_type=event_type_enum,
                callback=handler,
                filters=None,
                priority=5,
                async_handler=asyncio.iscoroutinefunction(handler)
            )
            
            # Add to subscribers list (sorted by priority)
            self.subscribers[event_type_enum].append(subscription)
            self.subscribers[event_type_enum].sort(key=lambda x: x.priority)
            
            # Store handler to subscriber ID mapping for unsubscribe
            if not hasattr(self, '_handler_to_subscriber'):
                self._handler_to_subscriber = {}
            self._handler_to_subscriber[(event_type, handler)] = subscriber_id
            
        except Exception as e:
            print(f"Subscribe error: {e}")
    
    async def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """
        Unsubscribe from events (async wrapper matching API spec)
        
        Args:
            event_type: Type of event as string
            handler: Handler function to unsubscribe
        """
        try:
            # Convert string to EventType enum
            if hasattr(EventType, event_type.upper()):
                event_type_enum = getattr(EventType, event_type.upper())
            else:
                # Try direct value match
                for et in EventType:
                    if et.value == event_type:
                        event_type_enum = et
                        break
                else:
                    print(f"Unknown event type: {event_type}")
                    return
            
            # Find and remove subscription with matching handler
            self.subscribers[event_type_enum] = [
                sub for sub in self.subscribers[event_type_enum]
                if sub.callback != handler
            ]
            
            # Clean up handler mapping if exists
            if hasattr(self, '_handler_to_subscriber'):
                key = (event_type, handler)
                if key in self._handler_to_subscriber:
                    del self._handler_to_subscriber[key]
                    
        except Exception as e:
            print(f"Unsubscribe error: {e}")
    
    async def process_events(self) -> None:
        """
        Process events from the queue (public wrapper matching API spec)
        """
        await self._process_events()
            
class EventLogger:
    """Event logger for persistence and analysis"""
    
    def __init__(self, log_file: str = "events.log"):
        """
        Initialize event logger
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        
    async def log_event(self, event: Event):
        """Log event to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(event.to_json() + '\n')
                
        except Exception as e:
            print(f"Event logging error: {e}")
            
    def read_events(self, limit: Optional[int] = None) -> List[Event]:
        """Read events from log file"""
        events = []
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                
                if limit:
                    lines = lines[-limit:]
                    
                for line in lines:
                    try:
                        data = json.loads(line.strip())
                        # Reconstruct event
                        event = Event(
                            event_type=EventType(data['event_type']),
                            data=data['data'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            event_id=data['event_id'],
                            source=data.get('source'),
                            priority=data.get('priority', 5),
                            metadata=data.get('metadata', {})
                        )
                        events.append(event)
                    except Exception as e:
                        print(f"Event parsing error: {e}")
                        
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Event reading error: {e}")
            
        return events
        
class EventAggregator:
    """Aggregate events for batch processing"""
    
    def __init__(self, batch_size: int = 100, timeout: float = 5.0):
        """
        Initialize event aggregator
        
        Args:
            batch_size: Maximum batch size
            timeout: Maximum time to wait for batch
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.batch: List[Event] = []
        self.last_flush = datetime.now()
        self.lock = asyncio.Lock()
        
    async def add_event(self, event: Event) -> Optional[List[Event]]:
        """
        Add event to batch
        
        Args:
            event: Event to add
            
        Returns:
            Batch if ready to process
        """
        async with self.lock:
            self.batch.append(event)
            
            # Check if batch is ready
            if len(self.batch) >= self.batch_size:
                return await self.flush()
                
            # Check timeout
            if (datetime.now() - self.last_flush).total_seconds() >= self.timeout:
                return await self.flush()
                
            return None
            
    async def flush(self) -> List[Event]:
        """Flush current batch"""
        async with self.lock:
            batch = self.batch.copy()
            self.batch.clear()
            self.last_flush = datetime.now()
            return batch