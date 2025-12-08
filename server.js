require('dotenv').config();
const express = require('express');
const axios = require('axios');
const Anthropic = require('@anthropic-ai/sdk');
const mongoose = require('mongoose');

const app = express();
app.use(express.json());

// Conectar MongoDB
mongoose.connect(process.env.MONGODB_URI);

// ============================================
// SCHEMAS DE BASE DE DATOS
// ============================================

const ClientSchema = new mongoose.Schema({
  clientId: { type: String, unique: true, required: true },
  nombre: String,
  greenApiInstance: String,
  greenApiToken: String,
  comercialWhatsApp: String,
  propiedades: [{
    id: Number,
    tipo: String,
    operacion: String,
    ubicacion: String,
    direccion: String,
    precio: Number,
    precioVenta: Number,
    precioAlquiler: Object,
    periodo: String,
    dormitorios: Number,
    banos: Number,
    superficie: Number,
    descripcion: String,
    caracteristicas: [String],
    url: String,
    urlInfocasas: String,
    referencia: String,
    referenciaInfocasas: String,
    disponible: { type: Boolean, default: true }
  }],
  activo: { type: Boolean, default: true },
  createdAt: { type: Date, default: Date.now }
});

const ConversationSchema = new mongoose.Schema({
  clientId: String,
  chatId: String,
  phoneNumber: String,
  messages: [{
    role: String,
    content: String,
    timestamp: { type: Date, default: Date.now }
  }],
  qualified: { type: Boolean, default: false },
  qualifiedAt: Date,
  notified: { type: Boolean, default: false },
  leadAnalysis: {
    operacion: String,
    tienePropiedadVista: Boolean,
    propiedadVista: String,
    linkCompartido: String,
    tipoPropiedad: String,
    zona: String,
    presupuesto: String,
    periodo: String,
    dormitorios: String,
    urgencia: String,
    propiedadesInteres: [String],
    resumen: String
  },
  createdAt: { type: Date, default: Date.now },
  lastActivity: { type: Date, default: Date.now }
});

ConversationSchema.index({ clientId: 1, chatId: 1 });
ConversationSchema.index({ clientId: 1, qualified: 1 });

const Client = mongoose.model('Client', ClientSchema);
const Conversation = mongoose.model('Conversation', ConversationSchema);

// ============================================
// CONFIGURACIÓN CLAUDE
// ============================================

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY
});

// ============================================
// FUNCIONES AUXILIARES
// ============================================

async function getClientByInstance(instanceId) {
  return await Client.findOne({ 
    greenApiInstance: instanceId,
    activo: true 
  });
}

async function sendWhatsAppMessage(client, chatId, message) {
  try {
    const url = `https://api.green-api.com/waInstance${client.greenApiInstance}/sendMessage/${client.greenApiToken}`;
    const response = await axios.post(url, {
      chatId: chatId,
      message: message
    });
    return response.data;
  } catch (error) {
    console.error('Error enviando mensaje:', error.message);
    throw error;
  }
}

function generateSystemPrompt(propiedades) {
  return `Eres un agente inmobiliario profesional de Fincas del Este, una prestigiosa inmobiliaria de Punta del Este, Uruguay.

PROPIEDADES DISPONIBLES:
${JSON.stringify(propiedades, null, 2)}

CAPACIDADES ESPECIALES - RECONOCIMIENTO DE LINKS:
- Podés reconocer links de: fincasdeleste.com.uy, infocasas.com.uy, mercadolibre.com.uy, gallito.com.uy
- Si el cliente comparte un link, decile "Déjame revisar ese link..." y luego:
  1. Identificá la propiedad por: ubicación, dormitorios, baños, superficie, descripción
  2. Buscá coincidencias en nuestra base de datos
  3. Si encontrás match, confirmá: "Perfecto! Es nuestra propiedad en [ubicación], Ref #[ref]. ¿Te interesa coordinar una visita?"
  4. Si NO encontrás match: "Esa propiedad no la manejamos, pero tengo opciones similares en la misma zona"

REGLA CRÍTICA - NO COMPARTIR DATOS DE CONTACTO:
- NUNCA des el número de teléfono de la inmobiliaria
- NUNCA des el email de contacto
- NUNCA des la dirección física
- Si preguntan "¿cómo los contacto?", respondé: "No te preocupes, un asesor te va a contactar por este mismo número en breve"

FLUJO DE CONVERSACIÓN:
1. Saludo inicial como asistente de Fincas del Este
2. Primera pregunta: "¿Estás buscando COMPRAR o ALQUILAR?"
   - Si alquilar: "¿Para TEMPORADA (enero/febrero) o ANUAL (contrato 2 años)?"
3. Segunda pregunta: "¿Ya tenés alguna propiedad vista (link, dirección) o estás buscando asesoramiento general?"
4. Si tiene propiedad vista: Identificar y ofrecer coordinar visita
5. Si busca asesoramiento: Calificar según operación
6. Cuando esté calificado: "Perfecto, te voy a conectar con uno de nuestros asesores"

PREGUNTAS DE CALIFICACIÓN:
- VENTAS: zona, presupuesto USD, tipo, dormitorios, para vivir/inversión
- ALQUILERES TEMPORARIOS: período, personas, zona, presupuesto por período, servicios
- ALQUILERES ANUALES: presupuesto mensual USD, zona, dormitorios, garaje, mascotas

IMPORTANTE:
- Respuestas BREVES (2-3 líneas)
- Lenguaje uruguayo natural
- NO compartir datos de contacto
- Formato precios: "U$S 350.000" (venta), "U$S 18.000 por Enero" (temporario), "U$S 1.200 por mes" (anual)`;
}

async function analyzeLeadInterest(conversationHistory, propiedades) {
  try {
    const analysisPrompt = `Analiza esta conversación inmobiliaria de Fincas del Este y extrae:
1. Operación: Venta/Alquiler Temporario/Alquiler Anual
2. ¿Tiene propiedad vista? (Sí/No)
3. Dirección o referencia vista
4. Link compartido (URL si hay)
5. Tipo de propiedad
6. Zona de interés
7. Presupuesto
8. Período (si es temporario)
9. Dormitorios
10. Urgencia

PROPIEDADES:
${JSON.stringify(propiedades, null, 2)}

CONVERSACIÓN:
${JSON.stringify(conversationHistory, null, 2)}

Respondé SOLO en formato JSON:
{
  "operacion": "...",
  "tienePropiedadVista": true/false,
  "propiedadVista": "...",
  "linkCompartido": "...",
  "tipoPropiedad": "...",
  "zona": "...",
  "presupuesto": "...",
  "periodo": "...",
  "dormitorios": "...",
  "urgencia": "...",
  "propiedadesInteres": [],
  "resumen": "..."
}`;

    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 600,
      messages: [{ role: 'user', content: analysisPrompt }]
    });

    const text = response.content[0].text;
    const clean = text.replace(/```json|```/g, '').trim();
    return JSON.parse(clean);
  } catch (error) {
    console.error('Error analizando lead:', error);
    return {
      operacion: "No especificado",
      tienePropiedadVista: false,
      propiedadVista: "",
      linkCompartido: "",
      tipoPropiedad: "No especificado",
      zona: "Punta del Este",
      presupuesto: "A consultar",
      periodo: "",
      dormitorios: "No especificado",
      urgencia: "media",
      propiedadesInteres: [],
      resumen: "Lead interesado en propiedades"
    };
  }
}

async function notifyComercial(client, conversation) {
  try {
    const analysis = conversation.leadAnalysis;
    
    let propiedadesDetalle = '';
    if (analysis.propiedadesInteres?.length > 0) {
      analysis.propiedadesInteres.forEach(ref => {
        const prop = client.propiedades.find(p => p.referencia === ref);
        if (prop) {
          propiedadesDetalle += `\n📍 ${prop.tipo} - ${prop.ubicacion}\n💰 ${formatPrice(prop)}\n🛏️ ${prop.dormitorios} dorm - ${prop.superficie}m²\n`;
        }
      });
    }

    const mensaje = `🎯 *NUEVO LEAD CALIFICADO - FINCAS DEL ESTE*

👤 *Cliente:* ${conversation.phoneNumber}
⏰ *Fecha:* ${new Date().toLocaleString('es-UY')}

📊 *PERFIL DEL CLIENTE:*
🎯 Operación: ${analysis.operacion}
${analysis.tienePropiedadVista ? `📍 Propiedad Vista: ${analysis.propiedadVista}` : ''}
${analysis.linkCompartido ? `🔗 Link: ${analysis.linkCompartido}` : ''}
${analysis.periodo ? `📅 Período: ${analysis.periodo}` : ''}
🏠 Tipo: ${analysis.tipoPropiedad}
📍 Zona: ${analysis.zona}
💵 Presupuesto: ${analysis.presupuesto}
🛏️ Dormitorios: ${analysis.dormitorios}
⚡ Urgencia: ${analysis.urgencia}

💡 *RESUMEN:*
${analysis.resumen}
${propiedadesDetalle ? '\n🏘️ *PROPIEDADES DE INTERÉS:*' + propiedadesDetalle : ''}

✅ *El lead está listo para coordinar ${analysis.tienePropiedadVista ? 'visita a la propiedad' : 'recorrida de opciones'}*

Link: https://wa.me/${conversation.phoneNumber.replace(/\D/g, '')}`;

    const comercialChatId = client.comercialWhatsApp.includes('@') 
      ? client.comercialWhatsApp 
      : `${client.comercialWhatsApp}@c.us`;

    await sendWhatsAppMessage(client, comercialChatId, mensaje);
    
    conversation.notified = true;
    await conversation.save();
    
    console.log(`✅ Notificación enviada a comercial de ${client.nombre}`);
    
  } catch (error) {
    console.error('Error notificando comercial:', error.message);
  }
}

function formatPrice(property) {
  if (property.operacion === 'Venta') {
    return `U$S ${property.precio.toLocaleString()}`;
  } else if (property.operacion === 'Alquiler Temporario') {
    return `U$S ${property.precio.toLocaleString()} por ${property.periodo}`;
  } else if (property.operacion === 'Venta y Alquiler Temporario') {
    return `Venta: U$S ${property.precioVenta.toLocaleString()} | Alquiler: desde U$S ${property.precioAlquiler.invernal}/mes`;
  } else {
    return `U$S ${property.precio.toLocaleString()} por mes`;
  }
}

// ============================================
// WEBHOOK PRINCIPAL
// ============================================

app.post('/webhook/:instanceId', async (req, res) => {
  try {
    const { instanceId } = req.params;
    const notification = req.body;
    
    const client = await getClientByInstance(instanceId);
    if (!client) {
      console.log(`Cliente no encontrado para instance: ${instanceId}`);
      return res.sendStatus(404);
    }

    if (notification.typeWebhook !== 'incomingMessageReceived') {
      return res.sendStatus(200);
    }

    const messageData = notification.messageData;
    
    if (messageData.typeMessage === 'outgoingMessageReceived') {
      return res.sendStatus(200);
    }

    const chatId = messageData.chatId;
    const phoneNumber = chatId.split('@')[0];
    const messageText = messageData.textMessageData?.textMessage;

    if (!messageText) {
      return res.sendStatus(200);
    }

    console.log(`📱 [${client.nombre}] Mensaje de ${phoneNumber}: ${messageText}`);

    let conversation = await Conversation.findOne({ 
      clientId: client.clientId, 
      chatId 
    });

    if (!conversation) {
      conversation = new Conversation({
        clientId: client.clientId,
        chatId,
        phoneNumber,
        messages: []
      });
    }

    conversation.messages.push({
      role: 'user',
      content: messageText
    });
    conversation.lastActivity = new Date();

    const systemPrompt = generateSystemPrompt(client.propiedades);
    const conversationHistory = [
      { role: 'user', content: systemPrompt },
      { role: 'assistant', content: 'Entendido.' },
      ...conversation.messages.map(m => ({
        role: m.role,
        content: m.content
      }))
    ];

    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 500,
      messages: conversationHistory
    });

    const aiResponse = response.content[0].text;

    conversation.messages.push({
      role: 'assistant',
      content: aiResponse
    });

    if (conversation.messages.length > 40) {
      conversation.messages = conversation.messages.slice(-40);
    }

    const lowerResponse = aiResponse.toLowerCase();
    const isQualified = (lowerResponse.includes('asesor') || 
                        lowerResponse.includes('vendedor') || 
                        lowerResponse.includes('conectar') ||
                        lowerResponse.includes('derivar') ||
                        lowerResponse.includes('visita')) && !conversation.qualified;

    if (isQualified) {
      conversation.qualified = true;
      conversation.qualifiedAt = new Date();
      
      const analysis = await analyzeLeadInterest(
        conversation.messages, 
        client.propiedades
      );
      conversation.leadAnalysis = analysis;
      
      await conversation.save();
      await notifyComercial(client, conversation);
    } else {
      await conversation.save();
    }

    await sendWhatsAppMessage(client, chatId, aiResponse);

    res.sendStatus(200);

  } catch (error) {
    console.error('Error en webhook:', error);
    res.sendStatus(500);
  }
});

// ============================================
// API DE ADMINISTRACIÓN
// ============================================

app.post('/admin/clients', async (req, res) => {
  try {
    const client = new Client(req.body);
    await client.save();
    res.json({ success: true, client });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get('/admin/clients', async (req, res) => {
  try {
    const clients = await Client.find({ activo: true });
    res.json(clients);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/admin/clients/:clientId', async (req, res) => {
  try {
    const client = await Client.findOne({ clientId: req.params.clientId });
    if (!client) {
      return res.status(404).json({ error: 'Cliente no encontrado' });
    }
    res.json(client);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/admin/clients/:clientId/conversations', async (req, res) => {
  try {
    const { qualified, limit = 50 } = req.query;
    
    const query = { clientId: req.params.clientId };
    if (qualified !== undefined) {
      query.qualified = qualified === 'true';
    }

    const conversations = await Conversation
      .find(query)
      .sort({ lastActivity: -1 })
      .limit(parseInt(limit));

    res.json({
      total: conversations.length,
      conversations
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/admin/clients/:clientId/stats', async (req, res) => {
  try {
    const { clientId } = req.params;
    
    const totalConversations = await Conversation.countDocuments({ clientId });
    const qualifiedLeads = await Conversation.countDocuments({ 
      clientId, 
      qualified: true 
    });
    
    const last7Days = new Date();
    last7Days.setDate(last7Days.getDate() - 7);
    
    const recentLeads = await Conversation.countDocuments({
      clientId,
      qualified: true,
      qualifiedAt: { $gte: last7Days }
    });

    res.json({
      totalConversations,
      qualifiedLeads,
      recentLeads,
      conversionRate: totalConversations > 0 
        ? ((qualifiedLeads / totalConversations) * 100).toFixed(2) 
        : 0
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/admin/dashboard', async (req, res) => {
  try {
    const totalClients = await Client.countDocuments({ activo: true });
    const totalConversations = await Conversation.countDocuments();
    const totalQualified = await Conversation.countDocuments({ qualified: true });
    
    const clientsWithStats = await Client.find({ activo: true });
    const stats = await Promise.all(
      clientsWithStats.map(async (client) => {
        const conversations = await Conversation.countDocuments({ 
          clientId: client.clientId 
        });
        const qualified = await Conversation.countDocuments({ 
          clientId: client.clientId, 
          qualified: true 
        });
        
        return {
          clientId: client.clientId,
          nombre: client.nombre,
          conversations,
          qualified,
          conversionRate: conversations > 0 
            ? ((qualified / conversations) * 100).toFixed(2) 
            : 0
        };
      })
    );

    res.json({
      totalClients,
      totalConversations,
      totalQualified,
      globalConversionRate: totalConversations > 0 
        ? ((totalQualified / totalConversations) * 100).toFixed(2) 
        : 0,
      clients: stats
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/', (req, res) => {
  res.json({
    service: 'PulseTrack - Agente Inmobiliario IA',
    client: 'Fincas del Este',
    status: 'running',
    version: '1.0.0',
    endpoints: {
      webhook: '/webhook/:instanceId',
      admin: '/admin/*'
    }
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`
  ✅ PulseTrack Server corriendo en puerto ${PORT}
  
  📊 MongoDB conectado
  🤖 Sistema listo para Fincas del Este
  
  Webhook URL:
  ${process.env.SERVER_URL || 'https://tu-servidor.com'}/webhook/{instanceId}
  `);
});
