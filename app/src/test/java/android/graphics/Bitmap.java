package android.graphics;

public class Bitmap {
    private final int width;
    private final int height;
    private final Config config;
    private final boolean isMutable;
    private boolean isRecycled = false;
    private final int[] pixels;

    public enum Config {
        ALPHA_8,
        RGB_565,
        ARGB_4444,
        ARGB_8888,
        RGBA_F16,
        HARDWARE
    }

    private Bitmap(int width, int height, Config config, boolean isMutable) {
        this.width = width;
        this.height = height;
        this.config = config;
        this.isMutable = isMutable;
        this.pixels = new int[width * height];
    }

    public static Bitmap createBitmap(int width, int height, Config config) {
        return new Bitmap(width, height, config, true);
    }

    public static Bitmap createBitmap(int width, int height) {
        return new Bitmap(width, height, Config.ARGB_8888, true);
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public Config getConfig() {
        return config;
    }

    public boolean isMutable() {
        return isMutable;
    }

    public boolean isRecycled() {
        return isRecycled;
    }

    public void recycle() {
        isRecycled = true;
    }

    public void setPixels(int[] pixels, int offset, int stride, int x, int y, int width, int height) {
        for (int row = 0; row < height; row++) {
            System.arraycopy(pixels, offset + row * stride, this.pixels, (y + row) * this.width + x, width);
        }
    }

    public void getPixels(int[] pixels, int offset, int stride, int x, int y, int width, int height) {
        for (int row = 0; row < height; row++) {
            System.arraycopy(this.pixels, (y + row) * this.width + x, pixels, offset + row * stride, width);
        }
    }

    public int getPixel(int x, int y) {
        return pixels[y * width + x];
    }

    public void setPixel(int x, int y, int color) {
        pixels[y * width + x] = color;
    }

    public Bitmap copy(Config config, boolean isMutable) {
        Bitmap copy = new Bitmap(width, height, config, isMutable);
        System.arraycopy(this.pixels, 0, copy.pixels, 0, this.pixels.length);
        return copy;
    }
}
