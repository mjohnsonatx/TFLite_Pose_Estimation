package android.os;

public final class SystemClock {
    private SystemClock() {}

    public static long uptimeMillis() {
        return System.nanoTime() / 1_000_000L;
    }

    public static long elapsedRealtime() {
        return System.nanoTime() / 1_000_000L;
    }

    public static long elapsedRealtimeNanos() {
        return System.nanoTime();
    }
}
