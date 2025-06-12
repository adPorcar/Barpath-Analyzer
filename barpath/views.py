from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
from ultralytics import YOLO
import io
import base64
from PIL import Image
import os
from pathlib import Path
from django.conf import settings

# Obtener la ruta absoluta del modelo YOLO
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'yolo11n.pt')

def home(request):
    return render(request, 'barpath/home.html')

def analysis(request):
    # Verificar si existe la imagen de trayectoria y el video procesado
    trajectory_path = os.path.join(settings.MEDIA_ROOT, 'trajectory_path.png')
    processed_video_path = os.path.join(settings.MEDIA_ROOT, 'processed_video.mp4')
    context = {}
    
    if os.path.exists(trajectory_path):
        with open(trajectory_path, 'rb') as f:
            plot_data = base64.b64encode(f.read()).decode('utf-8')
            context['plot'] = plot_data
    
    if os.path.exists(processed_video_path):
        # Obtener la URL relativa del video procesado para el template
        video_url = os.path.join(settings.MEDIA_URL, 'processed_video.mp4')
        context['video_url'] = video_url
    
    return render(request, 'barpath/analysis.html', context)

def gen_frames(video_path):
    """Genera frames del video con el análisis en tiempo real"""
    # Cargar el modelo YOLO11n
    model = YOLO(YOLO_MODEL_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    # Obtener información del video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    upper_zone = height * 0.4
    
    trajectory = []
    last_position = None
    min_confidence = 0.3
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir frame a gris para detección de movimiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is None:
            prev_frame = gray
            continue
        
        # Calcular diferencia entre frames
        frame_diff = cv2.absdiff(prev_frame, gray)
        prev_frame = gray.copy()
        
        # Umbralizar la diferencia
        _, motion_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
        
        # Ejecutar detección y tracking
        results = model.track(frame, persist=True, conf=min_confidence)
        
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                class_name = results[0].names[int(box.cls)]
                if class_name in ["frisbee", "clock"]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Verificar si hay movimiento en la región del objeto
                    roi = motion_mask[int(y1):int(y2), int(x1):int(x2)]
                    motion_detected = np.mean(roi) > 5
                    
                    if center_y < upper_zone or (last_position is not None and abs(center_y - last_position[1]) < 100):
                        if len(trajectory) == 0 or (
                            abs(center_x - trajectory[-1][0]) < 150 and
                            (last_position is None or abs(center_y - last_position[1]) < 100)
                        ) and (motion_detected or len(trajectory) == 0):
                            last_position = (center_x, center_y)
                            trajectory.append((center_x, center_y))
                            
                            # Dibujar el punto actual
                            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                            break
        
        # Dibujar la trayectoria completa
        if len(trajectory) > 1:
            # Suavizar la trayectoria con un filtro de media móvil
            if len(trajectory) > 3:
                smoothed = []
                for i in range(len(trajectory)):
                    start = max(0, i-2)
                    end = min(len(trajectory), i+3)
                    points = trajectory[start:end]
                    avg_x = int(sum(p[0] for p in points) / len(points))
                    avg_y = int(sum(p[1] for p in points) / len(points))
                    smoothed.append((avg_x, avg_y))
                
                # Dibujar la trayectoria suavizada
                for i in range(1, len(smoothed)):
                    cv2.line(frame, smoothed[i-1], smoothed[i], (0, 0, 255), 2)
            else:
                # Si hay pocos puntos, dibujar sin suavizar
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
        
        # Convertir frame a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@csrf_exempt
def video_feed(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        # Guardar el video temporalmente
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_video.mp4')
        with open(temp_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        
        return StreamingHttpResponse(gen_frames(temp_path),
                                  content_type='multipart/x-mixed-replace; boundary=frame')
    
    return JsonResponse({'error': 'No video file received'})

@csrf_exempt
def process_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        # Asegurarse de que existe el directorio media
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        
        video_file = request.FILES['video']
        # Guardar una copia permanente del video
        video_path = os.path.join(settings.MEDIA_ROOT, 'analyzed_video.mp4')
        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        
        # Cargar el modelo YOLO11n
        model = YOLO(YOLO_MODEL_PATH)
        
        # Procesar el video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JsonResponse({
                'success': False,
                'message': 'Error al abrir el video'
            })
        
        # Obtener información del video
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Configurar el video de salida
        processed_video_path = os.path.join(settings.MEDIA_ROOT, 'processed_video.mp4')
        # Intentar diferentes códecs compatibles con MP4
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("Error with avc1 codec")
        except:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'H264')  # Alternativa H.264
                out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    raise Exception("Error with H264 codec")
            except:
                # Último intento con codec básico
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
        
        trajectory = []
        last_position = None
        upper_zone = height * 0.4
        min_confidence = 0.3
        prev_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir frame a gris para detección de movimiento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is None:
                prev_frame = gray
                continue
            
            # Calcular diferencia entre frames
            frame_diff = cv2.absdiff(prev_frame, gray)
            prev_frame = gray.copy()
            
            # Umbralizar la diferencia
            _, motion_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
            
            # Ejecutar detección y tracking
            results = model.track(frame, persist=True, conf=min_confidence)
            
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    class_name = results[0].names[int(box.cls)]
                    if class_name in ["frisbee", "clock"]:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Verificar si hay movimiento en la región del objeto
                        roi = motion_mask[int(y1):int(y2), int(x1):int(x2)]
                        motion_detected = np.mean(roi) > 5
                        
                        if center_y < upper_zone or (last_position is not None and abs(center_y - last_position[1]) < 100):
                            if len(trajectory) == 0 or (
                                abs(center_x - trajectory[-1][0]) < 150 and
                                (last_position is None or abs(center_y - last_position[1]) < 100)
                            ) and (motion_detected or len(trajectory) == 0):
                                last_position = (center_x, center_y)
                                trajectory.append((center_x, center_y))
                                
                                # Dibujar el punto actual
                                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                                break
            
            # Dibujar la trayectoria completa
            if len(trajectory) > 1:
                # Suavizar la trayectoria con un filtro de media móvil
                if len(trajectory) > 3:
                    smoothed = []
                    for i in range(len(trajectory)):
                        start = max(0, i-2)
                        end = min(len(trajectory), i+3)
                        points = trajectory[start:end]
                        avg_x = int(sum(p[0] for p in points) / len(points))
                        avg_y = int(sum(p[1] for p in points) / len(points))
                        smoothed.append((avg_x, avg_y))
                    
                    # Dibujar la trayectoria suavizada
                    for i in range(1, len(smoothed)):
                        cv2.line(frame, smoothed[i-1], smoothed[i], (0, 0, 255), 2)
                else:
                    # Si hay pocos puntos, dibujar sin suavizar
                    for i in range(1, len(trajectory)):
                        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
            
            # Guardar el frame en el video procesado
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Crear imagen final con la trayectoria
        final_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        if len(trajectory) > 1:
            # Suavizar la trayectoria final
            if len(trajectory) > 3:
                smoothed = []
                for i in range(len(trajectory)):
                    start = max(0, i-2)
                    end = min(len(trajectory), i+3)
                    points = trajectory[start:end]
                    avg_x = int(sum(p[0] for p in points) / len(points))
                    avg_y = int(sum(p[1] for p in points) / len(points))
                    smoothed.append((avg_x, avg_y))
                
                # Dibujar la trayectoria suavizada
                for i in range(1, len(smoothed)):
                    cv2.line(final_image, smoothed[i-1], smoothed[i], (0, 0, 255), 2)
            else:
                # Si hay pocos puntos, dibujar sin suavizar
                for i in range(1, len(trajectory)):
                    cv2.line(final_image, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
            
            # Marcar inicio y fin
            cv2.circle(final_image, trajectory[0], 8, (0, 255, 0), -1)
            cv2.putText(final_image, "Inicio", (trajectory[0][0]+10, trajectory[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.circle(final_image, trajectory[-1], 8, (255, 0, 0), -1)
            cv2.putText(final_image, "Fin", (trajectory[-1][0]+10, trajectory[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Guardar la imagen
        output_path = os.path.join(settings.MEDIA_ROOT, 'trajectory_path.png')
        cv2.imwrite(output_path, final_image)
        
        # Convertir la imagen para mostrarla en el template
        with open(output_path, 'rb') as f:
            plot_data = base64.b64encode(f.read()).decode('utf-8')
        
        return JsonResponse({
            'success': True,
            'plot': plot_data,
            'message': 'Video procesado correctamente'
        })
    
    return JsonResponse({
        'success': False,
        'message': 'Error al procesar el video'
    })
