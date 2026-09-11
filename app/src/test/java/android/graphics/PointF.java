package android.graphics;

public class PointF {
    public float x;
    public float y;

    public PointF() {
    }

    public PointF(float x, float y) {
        this.x = x;
        this.y = y;
    }

    public PointF(PointF p) {
        if (p != null) {
            this.x = p.x;
            this.y = p.y;
        }
    }

    public void set(float x, float y) {
        this.x = x;
        this.y = y;
    }

    public final float length() {
        return (float) Math.hypot(x, y);
    }
}
