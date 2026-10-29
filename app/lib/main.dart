import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

/// Lado de la imagen que espera la red neuronal (128x128x3).
const int tamanoImagen = 128;

void main() => runApp(const AppReconocimiento());

class AppReconocimiento extends StatelessWidget {
  const AppReconocimiento({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Reconocimiento Facial',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF1E2761),
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: const Color(0xFF1E2761),
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      themeMode: ThemeMode.system,
      home: const PantallaPrincipal(),
    );
  }
}

class PantallaPrincipal extends StatefulWidget {
  const PantallaPrincipal({super.key});

  @override
  State<PantallaPrincipal> createState() => _EstadoPantallaPrincipal();
}

class _EstadoPantallaPrincipal extends State<PantallaPrincipal> {
  final ImagePicker _selectorImagen = ImagePicker();
  
  // URL de la API editable en vivo. Se puede inicializar con --dart-define=API_URL=...
  late String _urlApi;
  
  Uint8List? _bytesFoto;
  bool _cargando = false;
  bool _huboError = false;
  String _titulo = '';
  String _detalle = '';
  bool? _servidorConectado;

  @override
  void initState() {
    super.initState();
    // Determinamos un valor por defecto sensato según la plataforma
    const urlDefinida = String.fromEnvironment('API_URL');
    if (urlDefinida.isNotEmpty) {
      _urlApi = urlDefinida;
    } else if (Platform.isAndroid) {
      _urlApi = 'http://10.0.2.2:8000'; // Emulador Android
    } else {
      _urlApi = 'http://127.0.0.1:8000'; // iOS Simulator o macOS
    }
    _verificarEstadoServidor();
  }

  /// Verifica el endpoint /health para saber si la API está accesible
  Future<void> _verificarEstadoServidor() async {
    try {
      final respuesta = await http
          .get(Uri.parse('$_urlApi/health'))
          .timeout(const Duration(seconds: 4));
      if (mounted) {
        setState(() {
          _servidorConectado = (respuesta.statusCode == 200);
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _servidorConectado = false;
        });
      }
    }
  }

  void _mostrarResultado(String titulo, String detalle, {bool esError = false}) {
    setState(() {
      _titulo = titulo;
      _detalle = detalle;
      _huboError = esError;
    });
  }

  /// Captura desde cámara o galería y envía a inferencia
  Future<void> _obtenerImagenYConsultar(ImageSource origen) async {
    try {
      final XFile? foto = await _selectorImagen.pickImage(
        source: origen,
        maxWidth: 800,
        maxHeight: 800,
        imageQuality: 85,
      );
      if (foto == null) return; // cancelado por usuario

      final Uint8List bytesFoto = await foto.readAsBytes();
      setState(() {
        _bytesFoto = bytesFoto;
        _cargando = true;
        _titulo = '';
        _detalle = '';
      });

      // Procesamiento local: redimensionar a 128x128 y extraer bytes RGB
      final Uint8List matriz = _convertirAMatriz(bytesFoto);
      await _consultarApi(matriz);
    } on PlatformException catch (e) {
      _mostrarResultado('Permiso no concedido', 'Revisa los permisos de cámara/galería en los ajustes del dispositivo (${e.message}).', esError: true);
    } on FormatException catch (e) {
      _mostrarResultado('Imagen inválida', e.message, esError: true);
    } on TimeoutException {
      _mostrarResultado('Tiempo de espera agotado', 'El servidor tardó demasiado en responder. Revisa la URL y la conexión.', esError: true);
    } on SocketException {
      _mostrarResultado('Sin conexión', 'No se pudo contactar a "$_urlApi". Verifica que la API esté corriendo y accesible.', esError: true);
    } catch (e) {
      _mostrarResultado('Error inesperado', e.toString(), esError: true);
    } finally {
      if (mounted) setState(() => _cargando = false);
    }
  }

  /// Redimensiona a 128x128x3 y extrae matriz RGB plana uint8 (49152 bytes)
  Uint8List _convertirAMatriz(Uint8List bytesFoto) {
    final img.Image? imagenOriginal = img.decodeImage(bytesFoto);
    if (imagenOriginal == null) {
      throw const FormatException('No se pudo decodificar la imagen seleccionada');
    }
    final img.Image imagenOrientada = img.bakeOrientation(imagenOriginal);
    final img.Image imagenRedimensionada =
        img.copyResize(imagenOrientada, width: tamanoImagen, height: tamanoImagen);

    final Uint8List matriz = Uint8List(tamanoImagen * tamanoImagen * 3);
    int posicion = 0;
    for (int fila = 0; fila < tamanoImagen; fila++) {
      for (int columna = 0; columna < tamanoImagen; columna++) {
        final img.Pixel pixel = imagenRedimensionada.getPixel(columna, fila);
        matriz[posicion++] = pixel.r.toInt();
        matriz[posicion++] = pixel.g.toInt();
        matriz[posicion++] = pixel.b.toInt();
      }
    }
    return matriz;
  }

  /// Enviar matriz cruda en base64 al endpoint /predict
  Future<void> _consultarApi(Uint8List matriz) async {
    final http.Response respuesta = await http
        .post(
          Uri.parse('$_urlApi/predict'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'imagen_base64': base64Encode(matriz)}),
        )
        .timeout(const Duration(seconds: 45));

    final dynamic cuerpo = jsonDecode(utf8.decode(respuesta.bodyBytes));
    if (respuesta.statusCode != 200) {
      final String detalleError =
          (cuerpo is Map && cuerpo['detail'] != null) ? cuerpo['detail'].toString() : 'Error desconocido';
      _mostrarResultado('La API reportó un error', 'Código ${respuesta.statusCode}: $detalleError', esError: true);
      return;
    }

    final bool reconocido = cuerpo['reconocido'] as bool;
    final String porcentaje = ((cuerpo['confianza'] as num).toDouble() * 100).toStringAsFixed(1);
    if (reconocido) {
      _mostrarResultado(
        cuerpo['nombre'] as String,
        'Persona identificada con un $porcentaje% de confianza.',
      );
    } else {
      _mostrarResultado(
        'Persona desconocida',
        'No registrada en el dataset (confianza máxima: $porcentaje%).',
      );
    }
    _servidorConectado = true;
  }

  /// Modal para configurar la URL de la API fácilmente (iPhone, Android o Nube)
  void _mostrarDialogoConfiguracion() {
    final controlador = TextEditingController(text: _urlApi);
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Configurar API'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Ingresa la URL del backend FastAPI:',
                  style: TextStyle(fontSize: 13),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: controlador,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'URL de la API',
                    hintText: 'https://tu-api.onrender.com',
                  ),
                ),
                const SizedBox(height: 12),
                const Text('Atajos rápidos:', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
                Wrap(
                  spacing: 6,
                  children: [
                    ActionChip(
                      label: const Text('Render (Cloud)'),
                      onPressed: () => controlador.text = 'https://tu-servicio.onrender.com',
                    ),
                    ActionChip(
                      label: const Text('Android Emu (10.0.2.2)'),
                      onPressed: () => controlador.text = 'http://10.0.2.2:8000',
                    ),
                    ActionChip(
                      label: const Text('iOS Sim (127.0.0.1)'),
                      onPressed: () => controlador.text = 'http://127.0.0.1:8000',
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                const Text(
                  'Nota: En un iPhone o Android físico con Wi-Fi, usa la IP de tu PC (ej: http://192.168.1.15:8000) o un túnel ngrok.',
                  style: TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancelar'),
            ),
            FilledButton(
              onPressed: () {
                setState(() {
                  _urlApi = controlador.text.trim();
                  if (_urlApi.endsWith('/')) {
                    _urlApi = _urlApi.substring(0, _urlApi.length - 1);
                  }
                });
                Navigator.pop(context);
                _verificarEstadoServidor();
              },
              child: const Text('Guardar'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final ColorScheme colores = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Reconocimiento Facial'),
        actions: [
          // Indicador de conexión con el backend
          IconButton(
            tooltip: 'Estado de API y configuración',
            icon: Icon(
              Icons.circle,
              size: 14,
              color: _servidorConectado == true
                  ? Colors.greenAccent[700]
                  : (_servidorConectado == false ? Colors.redAccent : Colors.grey),
            ),
            onPressed: _mostrarDialogoConfiguracion,
          ),
          IconButton(
            tooltip: 'Ajustes de URL',
            icon: const Icon(Icons.settings),
            onPressed: _mostrarDialogoConfiguracion,
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
          child: Column(
            children: [
              // Banner de estado de conexión API
              GestureDetector(
                onTap: _mostrarDialogoConfiguracion,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: colores.surfaceVariant.withOpacity(0.5),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        _servidorConectado == true ? Icons.cloud_done : Icons.cloud_off,
                        size: 16,
                        color: _servidorConectado == true ? Colors.green : Colors.orange,
                      ),
                      const SizedBox(width: 8),
                      Flexible(
                        child: Text(
                          _urlApi,
                          style: TextStyle(fontSize: 12, color: colores.onSurfaceVariant),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      const SizedBox(width: 4),
                      const Icon(Icons.edit, size: 13),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // Área de vista previa de la foto
              Expanded(
                child: Center(
                  child: _bytesFoto == null
                      ? Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.face, size: 80, color: colores.outline),
                            const SizedBox(height: 12),
                            Text(
                              'Captura o selecciona una foto\npara identificar a la persona',
                              textAlign: TextAlign.center,
                              style: TextStyle(color: colores.outline, fontSize: 16),
                            ),
                          ],
                        )
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(16),
                          child: Image.memory(_bytesFoto!, fit: BoxFit.contain),
                        ),
                ),
              ),

              // Indicador de carga
              if (_cargando)
                const Padding(
                  padding: EdgeInsets.all(12),
                  child: Column(
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 8),
                      Text('Consultando modelo neuronal...', style: TextStyle(fontSize: 13)),
                    ],
                  ),
                ),

              // Tarjeta de resultado
              if (_titulo.isNotEmpty)
                Card(
                  elevation: 2,
                  color: _huboError ? colores.errorContainer : colores.primaryContainer,
                  child: ListTile(
                    leading: Icon(
                      _huboError ? Icons.error_outline : Icons.check_circle_outline,
                      size: 36,
                      color: _huboError ? colores.onErrorContainer : colores.onPrimaryContainer,
                    ),
                    title: Text(
                      _titulo,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 20,
                        color: _huboError ? colores.onErrorContainer : colores.onPrimaryContainer,
                      ),
                    ),
                    subtitle: Text(
                      _detalle,
                      style: TextStyle(
                        color: _huboError ? colores.onErrorContainer : colores.onPrimaryContainer,
                      ),
                    ),
                  ),
                ),

              const SizedBox(height: 16),

              // Botones de acción (Cámara y Galería)
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _cargando ? null : () => _obtenerImagenYConsultar(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Galería'),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 2,
                    child: FilledButton.icon(
                      onPressed: _cargando ? null : () => _obtenerImagenYConsultar(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Tomar foto'),
                      style: FilledButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
