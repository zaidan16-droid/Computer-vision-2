import cv2
import torch

class ObjectCounter:
    def __init__(self):
        # Load model lokal
        self.model = torch.load("model/yolov5s.pt", map_location="cpu")
        self.model.eval()
        self.names = self.model.names
        self.model.conf = 0.5

    def detect_and_count(self, frame):
        results = self.model(frame)
        df = results.pandas().xyxy[0]
        count = len(df)

        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
            cls = int(row['class'])
            label = self.names[cls]
            conf = row.confidence

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Overlay jumlah keseluruhan
        cv2.putText(frame, f"Total: {count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        return frame
